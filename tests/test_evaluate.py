"""Evaluate tests: the cross-product orchestration fills `results` and is idempotent.

Builds a tiny store in-test (one task, one program, one unit suite) and calls the
real evaluate (and the real score). A unit suite is used deliberately so grading
runs in-process via the worker with no Hypothesis subprocess, keeping the test
fast. Runs under pytest or standalone:

    python -m pytest tests/test_evaluate.py -q
    python tests/test_evaluate.py
"""
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbt import db
from pbt.core import Program, Suite, Task
from pbt.evaluate import evaluate


@contextmanager
def temp_db():
    """Yield a read-write connection to a throwaway database.

    Creates the schema in a temporary directory and tears it down on exit, so each
    test runs against an isolated store. A context manager rather than a pytest
    fixture so the file still runs standalone (see __main__).

    Yields:
        An open connection to a temporary SQLite database.
    """
    with tempfile.TemporaryDirectory() as d:
        conn = db.connect(os.path.join(d, "pbt.db"))
        try:
            yield conn
        finally:
            conn.close()


def _insert_task(conn: sqlite3.Connection, t: Task) -> None:
    """Insert a Task row, ignoring an existing one with the same id."""
    conn.execute(
        "INSERT OR IGNORE INTO tasks(task_id, dataset, prompt, reference_solution, setup, "
        "prelude, per_timeout, io_mode, difficulty) VALUES (?,?,?,?,?,?,?,?,?)",
        (t.task_id, t.dataset, t.prompt, t.reference_solution, t.setup, t.prelude,
         t.per_timeout, t.io_mode, t.difficulty),
    )


def _insert_program(conn: sqlite3.Connection, p: Program) -> None:
    """Insert a Program row, ignoring an existing one with the same id."""
    conn.execute(
        "INSERT OR IGNORE INTO programs(program_id, task_id, prog_model, code) VALUES (?,?,?,?)",
        (p.program_id, p.task_id, p.prog_model, p.code),
    )


def _insert_suite(conn: sqlite3.Connection, s: Suite) -> None:
    """Insert a Suite row, ignoring an existing one with the same id."""
    conn.execute(
        "INSERT OR IGNORE INTO suites(suite_id, task_id, suite_model, kind, code, valid) "
        "VALUES (?,?,?,?,?,?)",
        (s.suite_id, s.task_id, s.suite_model, s.kind, s.code, s.valid),
    )


def _seed_one(conn: sqlite3.Connection) -> tuple[Program, Suite]:
    """Seed one task, one program, and one passing unit suite into the store.

    The unit suite mirrors what pbt.seed writes: a JSON blob
    {"io_mode": "function", "tests": [...]} of assert strings the grading worker
    consumes directly. The asserts call the program's own top-level function so no
    prelude binding (e.g. `candidate`) is needed.

    Args:
        conn: Open read-write connection to the store.

    Returns:
        The inserted (program, suite) pair, for asserting on their result row.
    """
    _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="add two numbers"))
    program = Program(task_id="t1", prog_model="gpt-4", code="def add(a, b):\n    return a + b")
    suite_code = json.dumps(
        {"io_mode": "function", "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]},
        sort_keys=True,
    )
    suite = Suite(task_id="t1", suite_model="claude", kind="unit", code=suite_code)
    _insert_program(conn, program)
    _insert_suite(conn, suite)
    conn.commit()
    return program, suite


def test_evaluate_fills_results_for_the_cross_product():
    """A single program-by-suite pair produces exactly one results row."""
    with temp_db() as conn:
        program, suite = _seed_one(conn)
        n = evaluate(conn)
        assert n == 1, f"one program x one suite must score one pair, got {n}"
        rows = conn.execute(
            "SELECT passed, pass_fraction FROM results WHERE program_id = ? AND suite_id = ?",
            (program.program_id, suite.suite_id),
        ).fetchall()
        assert len(rows) == 1, "evaluate must write exactly one result row for the pair"
        assert rows[0]["passed"] == 1, "the correct program must pass its unit suite"
        assert rows[0]["pass_fraction"] == 1.0, "all unit cases pass, so pass_fraction is 1.0"


def test_evaluate_is_idempotent_on_rerun():
    """Re-running evaluate scores nothing new and leaves the results table unchanged."""
    with temp_db() as conn:
        _seed_one(conn)
        first = evaluate(conn)
        assert first == 1, f"first run must score the one pair, got {first}"
        (before,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        second = evaluate(conn)
        assert second == 0, f"re-run must score nothing already cached, got {second}"
        (after,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        assert before == after == 1, "the results table must be unchanged on re-run"


def test_evaluate_respects_dataset_filter():
    """Filtering to an absent dataset scores no pairs."""
    with temp_db() as conn:
        _seed_one(conn)  # dataset "mbppplus"
        n = evaluate(conn, dataset="humaneval")
        assert n == 0, f"no tasks in the filtered dataset means zero pairs, got {n}"
        (rows,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        assert rows == 0, "a non-matching dataset filter must write no results"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
