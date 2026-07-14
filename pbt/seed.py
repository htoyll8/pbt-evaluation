"""Seed the store from the existing `grading` dataset loaders.

Bridges the two `Task` representations: a grading `Task` (description + tests)
becomes one `pbt.core.Task` row in `tasks` plus one benchmark-authored
`pbt.core.Suite` of kind "unit" in `suites`. Both writes are INSERT OR IGNORE
keyed by content-hash IDs, so re-seeding the same dataset adds nothing new.

Unit-test serialization:
    The grading `Task.tests` field is exactly what the isolated worker
    (grading/evaluators/test_worker.py) consumes: a list of assert strings in
    "function" mode, or a list of [input, expected] pairs in "stdio" mode. The
    suite `code` blob is a JSON object {"io_mode": ..., "tests": [...]} so the
    suite is self-describing: a scorer can json.loads it, read the cases, and
    dispatch on io_mode without consulting any other table. The per-test
    execution context (setup, prelude, per_timeout) lives on the task row, where
    the worker already expects it.

Reference solutions and entry points:
    The grading loaders carry a full runnable `reference_solution` and an
    `entry_point` (the candidate function's name) onto their `Task` objects, and
    both are copied straight onto the seeded `tasks` row. The reference solution
    feeds the later PBT validity filter; the entry point lets score() bind the
    candidate by name. Loaders that name neither leave them "".
"""
import json
import sqlite3

from grading.datasets import load_tasks
from pbt.core import Suite, Task

BENCHMARK_MODEL = "benchmark"  # sentinel suite_model for the dataset's own unit tests


def _serialize_tests(io_mode: str, tests: list, prelude: str | None = None,
                     per_timeout: int | None = None) -> str:
    """Serialize a grading task's test units into a self-describing suite blob.

    Args:
        io_mode: "function" (assert strings) or "stdio" ([input, expected] pairs).
        tests: The grading `Task.tests` list, already worker-consumable.
        prelude: Suite-specific prelude overriding the task's, or None to inherit it.
        per_timeout: Suite-specific per-unit timeout overriding the task's, or None to
            inherit it.

    Returns:
        A compact JSON object {"io_mode": ..., "tests": [...]} usable directly
        by a scorer, carrying "prelude"/"per_timeout" only when the suite needs a
        context different from its task's. Omitting them keeps the blob (and therefore
        the suite's content-hash id) byte-identical to the pre-private-suite format.
    """
    blob: dict = {"io_mode": io_mode, "tests": tests}
    if prelude is not None:
        blob["prelude"] = prelude
    if per_timeout is not None:
        blob["per_timeout"] = per_timeout
    return json.dumps(blob, sort_keys=True)


def _insert_task(conn: sqlite3.Connection, task: Task) -> None:
    """Insert a task row, ignoring an existing row with the same id.

    Args:
        conn: Open read-write connection to the store.
        task: The normalized task to persist.
    """
    conn.execute(
        "INSERT OR IGNORE INTO tasks(task_id, dataset, prompt, reference_solution, "
        "entry_point, setup, prelude, per_timeout, io_mode, difficulty) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (task.task_id, task.dataset, task.prompt, task.reference_solution,
         task.entry_point, task.setup, task.prelude, task.per_timeout, task.io_mode,
         task.difficulty),
    )


def _insert_suite(conn: sqlite3.Connection, suite: Suite) -> None:
    """Insert a suite row, ignoring an existing row with the same id.

    Args:
        conn: Open read-write connection to the store.
        suite: The deduped test suite to persist.
    """
    conn.execute(
        "INSERT OR IGNORE INTO suites(suite_id, task_id, suite_model, kind, code, valid) "
        "VALUES (?,?,?,?,?,?)",
        (suite.suite_id, suite.task_id, suite.suite_model, suite.kind, suite.code,
         suite.valid),
    )


def seed_dataset(conn: sqlite3.Connection, dataset: str, n_tasks: int = 10,
                 difficulties: tuple[str, ...] | None = None) -> int:
    """Seed `tasks` and benchmark unit `suites` for a registered dataset.

    Loads up to `n_tasks` problems through the existing grading loaders, maps each
    onto a `pbt.core.Task`, and records the benchmark's own tests as a kind="unit"
    suite authored by the sentinel model "benchmark". Every write is an INSERT OR
    IGNORE on a content-hash key, so re-running seeds nothing new.

    Args:
        conn: Open read-write connection to the store.
        dataset: Name of a registered grading dataset (e.g. "mbppplus", "humaneval", "apps").
        n_tasks: Maximum number of tasks to load from the dataset.
        difficulties: Optional difficulty tiers to keep (only supported by "apps",
            e.g. ("introductory",) or ("competition",)); None loads the dataset as-is.

    Returns:
        The number of tasks processed from the loader (the count requested and
        mapped, independent of how many rows were newly inserted versus ignored).

    Raises:
        ValueError: If `dataset` is not a registered grading dataset.
    """
    grading_tasks = load_tasks(dataset, n_tasks, difficulties=difficulties)
    for gt in grading_tasks:
        task = Task(
            task_id=gt.task_id,
            dataset=dataset,
            prompt=gt.description,
            reference_solution=gt.reference_solution,
            entry_point=gt.entry_point,
            setup=gt.setup,
            prelude=gt.prelude,
            per_timeout=gt.per_timeout,
            io_mode=gt.io_mode,
            difficulty=gt.difficulty,
        )
        _insert_task(conn, task)
        _insert_suite(conn, Suite(
            task_id=task.task_id,
            suite_model=BENCHMARK_MODEL,
            kind="unit",
            code=_serialize_tests(gt.io_mode, gt.tests),
        ))
        if gt.private_tests:
            _insert_suite(conn, Suite(
                task_id=task.task_id,
                suite_model=BENCHMARK_MODEL,
                kind="private",
                code=_serialize_tests(gt.io_mode, gt.private_tests,
                                      prelude=gt.private_prelude,
                                      per_timeout=gt.private_timeout),
            ))
    conn.commit()
    return len(grading_tasks)
