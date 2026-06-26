"""Generate new candidate programs and PBTs from models into the store.

The seeding step (pbt.seed) records each benchmark task and its own unit tests.
This step asks models to produce the artifacts the study actually compares:
candidate programs (one `programs` row per distinct solution, one `generations`
row per sampling event) and property-based tests (one kind="pbt" `suites` row).

Both writers are INSERT OR IGNORE on content-hash IDs, so re-running the same
model over the same tasks adds nothing new. A task must already be seeded (the
programs/suites foreign keys point at `tasks`); generation only ever iterates
tasks already present in the store, so that constraint is satisfied by reading
the candidate tasks straight from the `tasks` table.

Model dependency injection:
    Both functions take an optional `model`. When it is None a real
    `model.Model` is constructed, but a caller (notably the tests) can pass any
    object exposing the single method this module depends on:

        complete(prompt: str) -> tuple[str, dict]

    returning the raw model text and a usage dict. That keeps the surface a stub
    must implement small and explicit, and keeps generation off the network in
    tests.

Reused prompts:
    The PBT prompt templates and the ```python``` code-block extractor are
    imported verbatim from scripts/generate_pbts.py, so the generated PBT honors
    the same `test_pbt(fn)` contract that pbt.core.score expects.

Attributes:
    PROGRAM_PROMPT (str): Template asking a model to solve a task and return its
        solution as a single triple-backtick Python block.
    PBT_VARIANTS (dict): Maps a variant name to its reused PBT prompt template.
"""
import json
import sqlite3
import sys
from pathlib import Path

import model as model_module
from pbt import ids
from pbt.core import Program, Suite, Task

# Reuse the existing generation prompts and the ```python``` extractor rather
# than duplicating them; the PBT templates already enforce the test_pbt(fn)
# contract that score() binds against.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_pbts import (  # noqa: E402  (path injected just above)
    PBT_PROMPT,
    PBT_PROMPT_NOHINTS,
    PBT_PROMPT_NOREF,
    PBT_PROMPT_NOREF_NOHINTS,
    extract_code,
)

PROGRAM_PROMPT = (
    "{prompt}\n\n"
    "Please provide a complete Python solution wrapped in a single "
    "triple-backtick block:\n"
    "```python\n# your code here\n```\n"
)

PBT_VARIANTS = {
    "ref": PBT_PROMPT,                       # task + reference solution + hints
    "noref": PBT_PROMPT_NOREF,               # no reference solution
    "nohints": PBT_PROMPT_NOHINTS,           # reference solution, no property hints
    "noref_nohints": PBT_PROMPT_NOREF_NOHINTS,
}


def _ensure_model(model, model_name: str):
    """Return the injected model, or build a real one for `model_name`.

    Args:
        model: A caller-supplied model exposing `complete(prompt) -> (text, usage)`,
            or None to construct a real `model.Model`.
        model_name: Name passed to `model.Model` when `model` is None.

    Returns:
        The model object to drive generation with.
    """
    if model is not None:
        return model
    return model_module.Model(model_name=model_name)


def _load_tasks(conn: sqlite3.Connection, dataset: str) -> list[Task]:
    """Load every seeded task for a dataset as `Task` objects.

    Args:
        conn: Open connection to the store.
        dataset: Dataset name to filter the `tasks` table on.

    Returns:
        The dataset's tasks ordered by id (empty if the dataset is unseeded).
    """
    rows = conn.execute(
        "SELECT task_id, dataset, prompt, reference_solution, entry_point, setup, "
        "prelude, per_timeout, io_mode, difficulty FROM tasks WHERE dataset = ? "
        "ORDER BY task_id",
        (dataset,),
    ).fetchall()
    return [
        Task(
            task_id=r["task_id"],
            dataset=r["dataset"],
            prompt=r["prompt"],
            reference_solution=r["reference_solution"] or "",
            entry_point=r["entry_point"] or "",
            setup=r["setup"] or "",
            prelude=r["prelude"] or "",
            per_timeout=r["per_timeout"],
            io_mode=r["io_mode"],
            difficulty=r["difficulty"],
        )
        for r in rows
    ]


def _unit_hints(conn: sqlite3.Connection, task_id: str) -> str:
    """Render a task's benchmark unit tests as prompt hint text.

    Reads the task's kind="unit" suite (the self-describing JSON blob written by
    pbt.seed) and formats up to three cases, mirroring the hint text the
    standalone PBT generator feeds its prompts.

    Args:
        conn: Open connection to the store.
        task_id: Task whose unit suite supplies the hints.

    Returns:
        Up to three formatted unit tests, or "" when the task has no unit suite.
    """
    row = conn.execute(
        "SELECT code FROM suites WHERE task_id = ? AND kind = 'unit' "
        "ORDER BY suite_id LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return ""
    blob = json.loads(row["code"])
    tests = blob.get("tests", [])[:3]
    if blob.get("io_mode") == "stdio":
        return "\n".join(
            f"Input:\n{inp}\nOutput:\n{out}" for inp, out in tests
        )
    return "\n".join(str(t) for t in tests)


def _insert_program(conn: sqlite3.Connection, program: Program) -> int:
    """Insert a program row, ignoring an existing one with the same id.

    Args:
        conn: Open read-write connection to the store.
        program: The candidate program to persist.

    Returns:
        1 if a new row was written, 0 if an identical program already existed.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO programs(program_id, task_id, prog_model, code) "
        "VALUES (?,?,?,?)",
        (program.program_id, program.task_id, program.prog_model, program.code),
    )
    return cur.rowcount


def _insert_generation(conn: sqlite3.Connection, task_id: str, prog_model: str,
                       sample_index: int, program_id: str) -> None:
    """Record one sampling event, ignoring a re-run of the same draw.

    Args:
        conn: Open read-write connection to the store.
        task_id: Task that was sampled.
        prog_model: Model that did the sampling.
        sample_index: Zero-based index of this draw within the task/model samples.
        program_id: Id of the program this draw produced.
    """
    conn.execute(
        "INSERT OR IGNORE INTO generations(gen_id, task_id, prog_model, "
        "sample_index, program_id) VALUES (?,?,?,?,?)",
        (ids.gen_id(task_id, prog_model, sample_index), task_id, prog_model,
         sample_index, program_id),
    )


def _insert_suite(conn: sqlite3.Connection, suite: Suite) -> int:
    """Insert a suite row, ignoring an existing one with the same id.

    Args:
        conn: Open read-write connection to the store.
        suite: The test suite to persist.

    Returns:
        1 if a new row was written, 0 if an identical suite already existed.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO suites(suite_id, task_id, suite_model, kind, code, "
        "valid) VALUES (?,?,?,?,?,?)",
        (suite.suite_id, suite.task_id, suite.suite_model, suite.kind, suite.code,
         suite.valid),
    )
    return cur.rowcount


def generate_programs(conn: sqlite3.Connection, dataset: str, prog_model: str,
                      n_samples: int = 1, model=None) -> int:
    """Generate candidate programs for every seeded task in a dataset.

    For each task, asks the model to solve `task.prompt` `n_samples` times,
    extracts the code from each response, and writes it as a `Program` plus one
    `generations` row per draw. Distinct draws that yield byte-identical code
    collapse to one program but keep separate generation events. All writes are
    idempotent, so re-running adds nothing new.

    Args:
        conn: Open read-write connection to the store.
        dataset: Dataset whose seeded tasks to solve.
        prog_model: Name of the model authoring the programs (also its provenance).
        n_samples: Number of programs to sample per task.
        model: Optional model exposing `complete(prompt) -> (text, usage)`; a real
            `model.Model(model_name=prog_model)` is built when None.

    Returns:
        The number of new program rows written (0 on a fully idempotent re-run).
    """
    client = _ensure_model(model, prog_model)
    tasks = _load_tasks(conn, dataset)
    if not tasks:
        print(f"[WARN] no seeded tasks for dataset {dataset!r}; nothing to generate",
              file=sys.stderr)
        return 0

    written = 0
    prompt_template = PROGRAM_PROMPT
    for task in tasks:
        prompt = prompt_template.format(prompt=task.prompt)
        for sample_index in range(n_samples):
            text, _ = client.complete(prompt)
            code = extract_code(text)
            program = Program(task_id=task.task_id, prog_model=prog_model, code=code)
            written += _insert_program(conn, program)
            _insert_generation(conn, task.task_id, prog_model, sample_index,
                               program.program_id)
    conn.commit()
    return written


def generate_pbts(conn: sqlite3.Connection, dataset: str, suite_model: str,
                  model=None, variant: str = "ref") -> int:
    """Generate one property-based test per seeded task in a dataset.

    Builds the reused PBT prompt for the chosen variant from `task.prompt`, the
    task's `reference_solution` (when the variant uses it), and the task's unit
    tests as hints, calls the model, extracts the ```python``` block, and writes
    it verbatim as a kind="pbt" `Suite`. Writes are idempotent on the suite's
    content hash.

    Args:
        conn: Open read-write connection to the store.
        dataset: Dataset whose seeded tasks to test.
        suite_model: Name of the model authoring the PBTs (also its provenance).
        model: Optional model exposing `complete(prompt) -> (text, usage)`; a real
            `model.Model(model_name=suite_model)` is built when None.
        variant: Which reused prompt to use, one of PBT_VARIANTS ("ref", "noref",
            "nohints", "noref_nohints").

    Returns:
        The number of new PBT suite rows written (0 on a fully idempotent re-run).

    Raises:
        ValueError: If `variant` is not a known PBT prompt variant.
    """
    if variant not in PBT_VARIANTS:
        raise ValueError(
            f"unknown PBT variant {variant!r}; choose from {sorted(PBT_VARIANTS)}"
        )
    template = PBT_VARIANTS[variant]
    client = _ensure_model(model, suite_model)
    tasks = _load_tasks(conn, dataset)
    if not tasks:
        print(f"[WARN] no seeded tasks for dataset {dataset!r}; nothing to generate",
              file=sys.stderr)
        return 0

    written = 0
    for task in tasks:
        fmt_kwargs = {"prompt": task.prompt, "unit_tests": _unit_hints(conn, task.task_id)}
        if "{code}" in template:  # ref/nohints variants want the reference solution
            fmt_kwargs["code"] = task.reference_solution
        text, _ = client.complete(template.format(**fmt_kwargs))
        code = extract_code(text)
        suite = Suite(task_id=task.task_id, suite_model=suite_model, kind="pbt",
                      code=code)
        written += _insert_suite(conn, suite)
    conn.commit()
    return written
