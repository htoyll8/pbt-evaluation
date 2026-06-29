"""Isolated, killable subprocess scorer.

Runs a candidate program's tests inside grading/evaluators/test_worker.py in its own process
group, so one infinite-loop (or crashing) candidate can't wedge the parent: a hard SIGKILL
backstop on the whole tree, and a crash kills only that subprocess. `make_scorer` returns
the pass fraction; `make_cases` returns the per-case pass vector.
"""
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_worker.py")

# Hard wall-clock ceiling per scorer call, independent of test count. Without it, a
# candidate that hangs on every one of N cases burns per_timeout*N seconds (many minutes
# for MBPP+'s ~150-case harnesses). Set generously (5 min) so a slow-but-correct solution
# is never falsely failed; it only fires on genuinely stuck candidates.
MAX_WALL_SECONDS = 300


def _budget(per_timeout: int, n_tests: int) -> tuple[int, int]:
    """(parent overall timeout, worker self-deadline). The worker stops at the deadline and
    prints partial progress, so hitting the wall cap yields partial credit instead of the 0.0
    we'd get if the parent SIGKILLed it first. The margin exceeds one per_timeout because the
    deadline is only checked between cases."""
    overall = min(per_timeout * n_tests + 15, MAX_WALL_SECONDS)
    return overall, max(overall - (per_timeout + 3), per_timeout)


def _invoke(base: dict, overall: int, code: str) -> dict | None:
    """Run the worker on one program; return its parsed sentinel dict, or None on
    timeout/crash/missing-sentinel."""
    payload = json.dumps({**base, "program": code})
    proc = subprocess.Popen(
        [sys.executable, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, preexec_fn=os.setsid,
    )
    try:
        out, _ = proc.communicate(payload, timeout=overall)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # kill the whole tree
        except Exception:
            proc.kill()
        return None
    for line in reversed(out.splitlines()):
        if line.startswith("##LCT##"):
            try:
                return json.loads(line[len("##LCT##"):])
            except Exception:
                return None
    return None


def make_scorer(setup: str, tests: list, prelude: str = "", per_timeout: int = 5,
                io_mode: str = "function") -> Callable[[str], float]:
    """Build a pass-fraction scorer for one task. `prelude` runs after the program (e.g. bind
    `candidate` for HumanEval). `io_mode` is "function" (assert tests) or "stdio" ([input,
    output] pairs via piped stdin). The returned callable scores a program in [0, 1]."""
    tests = list(tests)
    if not tests:
        return lambda code: 0.0
    overall, budget = _budget(per_timeout, len(tests))
    base = {"setup": setup or "", "prelude": prelude, "tests": tests,
            "per_timeout": per_timeout, "io_mode": io_mode, "budget": budget}

    def score(code: str) -> float:
        r = _invoke(base, overall, code)
        if not r or not r.get("total"):
            return 0.0
        return r["passed"] / r["total"]

    return score


def make_cases(setup: str, tests: list, prelude: str = "", per_timeout: int = 5,
               io_mode: str = "function") -> Callable[[str], list[bool]]:
    """Like make_scorer but returns the per-case pass vector (length len(tests)); an
    all-False vector on timeout/crash. Used for per-case analyses (e.g. regression audit)."""
    tests = list(tests)
    if not tests:
        return lambda code: []
    overall, budget = _budget(per_timeout, len(tests))
    base = {"setup": setup or "", "prelude": prelude, "tests": tests,
            "per_timeout": per_timeout, "io_mode": io_mode, "budget": budget}

    def cases(code: str) -> list[bool]:
        r = _invoke(base, overall, code)
        vec = r.get("cases") if r else None
        return list(vec) if vec is not None else [False] * len(tests)

    return cases
