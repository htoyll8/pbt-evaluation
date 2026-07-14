"""The one graded operation: score(program, suite) -> Result.

Every measurement in the study reduces to scoring one candidate program against
one test suite. This module owns that operation and nothing else:

1. Cache first. The `results` primary key (program_id, suite_id) is the cache, so
   a hit returns the stored Result without recomputing.
2. Dispatch on suite kind. "unit" and "private" suites reuse the existing isolated
   subprocess scorer (grading/evaluators); "pbt" suites run a Hypothesis property
   in an equally isolated subprocess.
3. Write once. The fresh Result is persisted with INSERT OR IGNORE (the first
   verdict wins, matching the append-only store) and returned.

Unit / private suites:
    A unit or private suite's `code` is the self-describing blob written by
    pbt.seed: a JSON object {"io_mode": ..., "tests": [...]} whose `tests` list is
    exactly what grading/evaluators/test_worker.py consumes (assert strings in
    "function" mode, [input, expected] pairs in "stdio" mode). We json.loads the
    blob, read its io_mode and tests, and hand them straight to make_scorer with
    the per-test execution context (setup, prelude, per_timeout) carried on the
    task row. pass_fraction is the worker's pass fraction; passed means it reached
    1.0. Partial credit therefore reflects however many cases the blob holds.

PBT suites:
    A pbt suite's `code` is a Hypothesis module defining `test_pbt(fn)` (a @given
    property that takes the candidate as its single argument), exactly as the
    pbt_data generators write it. We exec the program, bind the candidate by the
    task's `entry_point` when that name is set and present in the program (falling
    back to the first top-level `def` otherwise), and call test_pbt(candidate)
    inside an isolated
    subprocess that mirrors the scorer's process-group isolation and wall-clock
    SIGKILL backstop. passed is True when no counterexample is found; pass_fraction
    is binary (1.0 or 0.0) because a property has no partial credit.

Attributes:
    PBT_MAX_EXAMPLES (int): Hypothesis examples generated per property.
    PBT_DERANDOMIZE (bool): Pin a fixed, reproducible RNG seed (Hypothesis
        derandomize) so a verdict is deterministic across runs.
    PBT_DEADLINE: Per-example deadline, disabled (None) so a slow-but-correct
        program is never failed for timing.
    PBT_DATABASE: Hypothesis example database, disabled (None) so no counterexample
        is reused across runs and each verdict stands alone.
    PBT_PHASES (tuple): Verdict-only phases: generate fresh examples and run any
        explicit ones, but skip reuse, targeting, shrinking, and explaining since
        only the binary verdict is recorded.
    PBT_WALL_SECONDS (int): Hard wall-clock ceiling for one PBT subprocess; on
        expiry the process tree is SIGKILLed and the verdict is a fail.
"""
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Callable

from grading.evaluators.scorer import make_scorer
from pbt.core.types import Program, Result, Suite, Task

# PBT scoring contract. Module-level so the exact settings behind every verdict are
# easy to log and reproduce later. They pin Hypothesis to a deterministic, verdict-
# only run: a fixed seed, no deadline, no example-database reuse, no shrinking.
PBT_MAX_EXAMPLES = 100
PBT_DERANDOMIZE = True
PBT_DEADLINE = None
PBT_DATABASE = None
PBT_PHASES = ("explicit", "generate")
PBT_WALL_SECONDS = 120

# Worker source for the isolated PBT subprocess. Reads one JSON job on stdin and
# prints "##PBT##" + {"passed": bool}. Kept as a -c payload (not a separate file)
# so scoring stays self-contained; isolation comes from the parent's process group
# and wall-clock kill, exactly as grading/evaluators/scorer.py does it.
_PBT_WORKER = r'''
import json
import re
import sys


def _candidate_name(code):
    m = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else "candidate"


def _bind_candidate(namespace, code, entry_point):
    if entry_point and entry_point in namespace:
        return namespace[entry_point]  # the task names its function: bind it directly
    return namespace[_candidate_name(code)]  # fall back to the first top-level def


def main():
    job = json.load(sys.stdin)
    from hypothesis import HealthCheck, Phase, settings
    settings.register_profile(
        "pbt_score",
        max_examples=job["max_examples"],
        deadline=job["deadline"],
        database=job["database"],
        derandomize=job["derandomize"],
        phases=tuple(getattr(Phase, p) for p in job["phases"]),
        print_blob=False,
        suppress_health_check=list(HealthCheck),
    )
    settings.load_profile("pbt_score")

    namespace = {}
    passed = False
    try:
        exec("import sys\nfrom typing import *", namespace)
        exec(job["program"], namespace)
        # Bind the candidate BEFORE running the PBT: a top-level def in the PBT (e.g. a
        # reimplementation of the function under test) would otherwise overwrite the
        # candidate in this shared namespace, and we would test the PBT's own version.
        candidate = _bind_candidate(namespace, job["program"], job.get("entry_point", ""))
        exec(job["pbt"], namespace)
        namespace["test_pbt"](candidate)
        passed = True  # property held: no counterexample
    except BaseException:
        passed = False  # counterexample found (or the program/property errored)
    print("##PBT##" + json.dumps({"passed": passed}))


if __name__ == "__main__":
    main()
'''


def _cached_result(conn: sqlite3.Connection, program_id: str, suite_id: str) -> Result | None:
    """Return the stored Result for (program_id, suite_id), or None if uncached.

    Args:
        conn: Open connection to the store.
        program_id: ID of the scored program.
        suite_id: ID of the suite it was scored against.

    Returns:
        The cached Result if a row exists, otherwise None.
    """
    row = conn.execute(
        "SELECT program_id, suite_id, task_id, prog_model, suite_model, kind, passed, "
        "pass_fraction FROM results WHERE program_id = ? AND suite_id = ?",
        (program_id, suite_id),
    ).fetchone()
    if row is None:
        return None
    return Result(
        program_id=row["program_id"],
        suite_id=row["suite_id"],
        task_id=row["task_id"],
        prog_model=row["prog_model"],
        suite_model=row["suite_model"],
        kind=row["kind"],
        passed=bool(row["passed"]),
        pass_fraction=row["pass_fraction"],
    )


def _score_units(program: Program, suite: Suite, task: Task) -> tuple[bool, float]:
    """Score a unit or private suite via the isolated grading worker.

    The suite blob is self-describing JSON ({"io_mode": ..., "tests": [...]}); its
    tests list is fed verbatim to make_scorer. Execution context comes from the task
    unless the blob overrides it: a task's weak and private suites can need different
    contexts (MBPP+'s plus harness needs its preamble bound as a prelude, HumanEval's
    needs a longer timeout because it runs whole), and context lives on the task row.
    A blob without these keys behaves exactly as before.

    Args:
        program: The candidate program (only its `code` is run).
        suite: The unit or private suite whose JSON blob holds the test cases, and
            optionally its own prelude / per_timeout.
        task: The task carrying the default setup, prelude, and per_timeout.

    Returns:
        A (passed, pass_fraction) pair, where passed means every case passed.
    """
    blob = json.loads(suite.code)
    io_mode = blob.get("io_mode", task.io_mode)
    scorer: Callable[[str], float] = make_scorer(
        setup=task.setup,
        tests=blob["tests"],
        prelude=blob.get("prelude", task.prelude),
        per_timeout=blob.get("per_timeout", task.per_timeout),
        io_mode=io_mode,
    )
    fraction = scorer(program.code)
    return fraction >= 1.0, fraction


def run_property(program_code: str, pbt_code: str, entry_point: str = "",
                 per_timeout: int = 5) -> bool:
    """Run a PBT property against a program in an isolated subprocess.

    The public, DB-free core of PBT scoring. It runs the property module's
    `test_pbt(fn)` against the given program in the same isolated worker that
    `_score_pbt` uses (its own process group, with a wall-clock SIGKILL backstop),
    binding the candidate by entry_point and falling back to the first top-level
    def when entry_point is unset or absent. This neither reads nor writes the
    store, so it can validate arbitrary code, for example a task's reference
    solution against a candidate PBT.

    Args:
        program_code: Source of the program whose function the property is bound to.
        pbt_code: A Hypothesis module defining `test_pbt(fn)`.
        entry_point: Name of the function to bind, or "" to use the first def.
        per_timeout: Per-test budget that raises the wall-clock floor when larger
            than the module default.

    Returns:
        True if the property held with no counterexample, False if a counterexample
        was found or the program/property crashed, timed out, or failed to import.
    """
    job = json.dumps({
        "program": program_code,
        "pbt": pbt_code,
        "entry_point": entry_point,
        "max_examples": PBT_MAX_EXAMPLES,
        "derandomize": PBT_DERANDOMIZE,
        "deadline": PBT_DEADLINE,
        "database": PBT_DATABASE,
        "phases": list(PBT_PHASES),
    })
    wall = max(PBT_WALL_SECONDS, per_timeout)
    proc = subprocess.Popen(
        [sys.executable, "-c", _PBT_WORKER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, preexec_fn=os.setsid,
    )
    try:
        out, _ = proc.communicate(job, timeout=wall)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # kill the whole tree
        except Exception:
            proc.kill()
        return False
    for line in reversed(out.splitlines()):
        if line.startswith("##PBT##"):
            try:
                return bool(json.loads(line[len("##PBT##"):])["passed"])
            except Exception:
                return False
    return False  # no sentinel: crash or import failure counts as a fail


def _score_pbt(program: Program, suite: Suite, task: Task) -> tuple[bool, float]:
    """Score a PBT suite by running its property in an isolated subprocess.

    Mirrors the grading scorer's isolation: the worker runs in its own process
    group and the whole tree is SIGKILLed if it outlives the wall-clock ceiling.
    Delegates the actual run to run_property and maps the binary verdict onto a
    (passed, pass_fraction) pair.

    Args:
        program: The candidate program; the property is bound to the function
            named by the task's entry_point, or its first top-level def otherwise.
        suite: The PBT suite whose `code` defines `test_pbt(fn)`.
        task: The task, supplying entry_point for binding and per_timeout as the
            floor on the wall-clock budget.

    Returns:
        A (passed, pass_fraction) pair; passed means no counterexample was found
        and pass_fraction is binary (1.0 or 0.0).
    """
    passed = run_property(program.code, suite.code, task.entry_point, task.per_timeout)
    return passed, 1.0 if passed else 0.0


def score(conn: sqlite3.Connection, program: Program, suite: Suite, task: Task) -> Result:
    """Score one program against one suite, caching the verdict.

    The single graded operation of the study. Idempotent: the result is keyed by
    (program_id, suite_id) in the append-only store, so a cache hit returns
    immediately and a recomputed result never overwrites an existing one.

    Args:
        conn: Open read-write connection to the store.
        program: The candidate program to score.
        suite: The test suite to score it against ("unit", "private", or "pbt").
        task: The task both share, supplying the execution context (setup,
            prelude, per_timeout, io_mode).

    Returns:
        The Result for (program, suite): the cached row if present, otherwise the
        freshly computed and persisted verdict.

    Raises:
        ValueError: If program, suite, and task do not all share the same task_id.
    """
    if not (program.task_id == suite.task_id == task.task_id):
        raise ValueError(
            f"task_id mismatch: program={program.task_id!r} suite={suite.task_id!r} "
            f"task={task.task_id!r}"
        )

    cached = _cached_result(conn, program.program_id, suite.suite_id)
    if cached is not None:
        return cached

    if suite.kind == "pbt":
        passed, fraction = _score_pbt(program, suite, task)
    else:  # "unit" or "private": both are worker-consumable test blobs
        passed, fraction = _score_units(program, suite, task)

    result = Result(
        program_id=program.program_id,
        suite_id=suite.suite_id,
        task_id=task.task_id,
        prog_model=program.prog_model,
        suite_model=suite.suite_model,
        kind=suite.kind,
        passed=passed,
        pass_fraction=fraction,
    )
    conn.execute(
        "INSERT OR IGNORE INTO results(program_id, suite_id, task_id, prog_model, "
        "suite_model, kind, passed, pass_fraction) VALUES (?,?,?,?,?,?,?,?)",
        (result.program_id, result.suite_id, result.task_id, result.prog_model,
         result.suite_model, result.kind, int(result.passed), result.pass_fraction),
    )
    conn.commit()
    return result
