"""Seeding tests: tasks land in `tasks`, each gets a unit suite, re-runs no-op.

Fast and offline: the grading loader is stubbed so no HF dataset is downloaded.
Runs under pytest or standalone:

    python -m pytest tests/test_seed.py -q
    python tests/test_seed.py
"""
import json
import os
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grading.datasets.base import Task as GradingTask
from pbt import db, seed


@contextmanager
def temp_db():
    """Yield a read-write connection to a throwaway database.

    Creates the schema in a temporary directory and tears it down on exit, so each
    test runs against an isolated store. Deliberately a context manager rather than
    a pytest fixture so the file still runs standalone (see __main__).

    Yields:
        An open connection to a temporary SQLite database.
    """
    with tempfile.TemporaryDirectory() as d:
        conn = db.connect(os.path.join(d, "pbt.db"))
        try:
            yield conn
        finally:
            conn.close()


def _fake_loader(tasks: list) -> "callable":
    """Build a `load_tasks` stand-in returning a fixed slice of grading tasks.

    Args:
        tasks: Grading `Task` objects to serve.

    Returns:
        A function with the `load_tasks(name, n_tasks)` signature.
    """
    def loader(name: str, n_tasks: int) -> list:
        return tasks[:n_tasks]
    return loader


def _function_tasks() -> list:
    """Two function-mode grading tasks with assert-string tests."""
    return [
        GradingTask(task_id="f1", description="add", setup="", tests=["assert add(1,2)==3"],
                    prelude="candidate = add"),
        GradingTask(task_id="f2", description="sub", setup="", tests=["assert sub(2,1)==1"]),
    ]


def _stdio_task() -> GradingTask:
    """One stdio-mode grading task with an [input, expected] pair."""
    return GradingTask(task_id="s1", description="echo", setup="", tests=[["hi\n", "hi\n"]],
                       io_mode="stdio")


def test_seed_writes_expected_task_count(monkeypatch):
    """Seeding writes one `tasks` row per loaded grading task and returns that count."""
    monkeypatch.setattr(seed, "load_tasks", _fake_loader(_function_tasks()))
    with temp_db() as conn:
        n = seed.seed_dataset(conn, "mbppplus", n_tasks=10)
        (rows,) = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
        assert n == 2, f"expected 2 tasks loaded, got {n}"
        assert rows == 2, f"expected 2 task rows, got {rows}"


def test_seed_respects_n_tasks(monkeypatch):
    """The n_tasks cap limits how many tasks are seeded."""
    monkeypatch.setattr(seed, "load_tasks", _fake_loader(_function_tasks()))
    with temp_db() as conn:
        n = seed.seed_dataset(conn, "mbppplus", n_tasks=1)
        (rows,) = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
        assert n == 1 and rows == 1, f"n_tasks=1 must seed exactly one task, got n={n} rows={rows}"


def test_each_task_gets_a_unit_suite(monkeypatch):
    """Every seeded task gets exactly one kind='unit' suite authored by 'benchmark'."""
    monkeypatch.setattr(seed, "load_tasks", _fake_loader(_function_tasks()))
    with temp_db() as conn:
        seed.seed_dataset(conn, "mbppplus", n_tasks=10)
        rows = conn.execute(
            "SELECT task_id, suite_model, kind FROM suites WHERE kind='unit'").fetchall()
        assert len(rows) == 2, f"expected one unit suite per task, got {len(rows)}"
        for r in rows:
            assert r["suite_model"] == "benchmark", f"unit suite must be benchmark-authored, got {r['suite_model']!r}"
        seeded = {r["task_id"] for r in rows}
        assert seeded == {"f1", "f2"}, f"every task must get a unit suite, got {seeded}"


def test_unit_suite_blob_is_consumable_json(monkeypatch):
    """The unit suite `code` blob is JSON carrying io_mode and the worker-ready tests."""
    monkeypatch.setattr(seed, "load_tasks", _fake_loader([_stdio_task()]))
    with temp_db() as conn:
        seed.seed_dataset(conn, "apps", n_tasks=10)
        (code,) = conn.execute(
            "SELECT code FROM suites WHERE task_id='s1' AND kind='unit'").fetchone()
        blob = json.loads(code)
        assert blob["io_mode"] == "stdio", f"blob must carry io_mode, got {blob.get('io_mode')!r}"
        assert blob["tests"] == [["hi\n", "hi\n"]], f"blob must carry the tests verbatim, got {blob['tests']!r}"


def test_seeding_is_idempotent(monkeypatch):
    """Re-running the same seed adds no new `tasks` or `suites` rows."""
    monkeypatch.setattr(seed, "load_tasks", _fake_loader(_function_tasks()))
    with temp_db() as conn:
        seed.seed_dataset(conn, "mbppplus", n_tasks=10)
        seed.seed_dataset(conn, "mbppplus", n_tasks=10)
        (tasks,) = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
        (suites,) = conn.execute("SELECT COUNT(*) FROM suites").fetchone()
        assert tasks == 2, f"re-seeding must not duplicate tasks, got {tasks}"
        assert suites == 2, f"re-seeding must not duplicate suites, got {suites}"


def _run_standalone() -> None:
    """Run every test with a minimal monkeypatch shim, no pytest required."""
    class _Patch:
        """Tiny monkeypatch stand-in supporting setattr with teardown."""

        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, value):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        patch = _Patch()
        try:
            fn(patch)
        finally:
            patch.undo()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_standalone()
