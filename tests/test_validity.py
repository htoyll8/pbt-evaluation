"""Validity-filter tests: does the reference solution pass each pbt?

Runs the real Hypothesis subprocess (hypothesis is installed) over a throwaway
store: a correct pbt the reference passes earns valid=1, a buggy pbt the reference
fails earns valid=0, and a task with no reference_solution is skipped (valid stays
NULL). Runs under pytest or standalone:

    python -m pytest tests/test_validity.py -q
    python tests/test_validity.py
"""
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbt import db, validity
from pbt.core import Suite, Task

# A correct property for `add`: the reference must satisfy it, so it is valid.
PBT_ADD_CORRECT = (
    "from hypothesis import given\n"
    "from hypothesis import strategies as st\n\n"
    "def test_pbt(fn):\n"
    "    @given(a=st.integers(), b=st.integers())\n"
    "    def _t(a, b):\n"
    "        assert fn(a, b) == a + b\n"
    "    _t()\n"
)

# A buggy property asserting something false (sum is always positive). Even the
# correct reference fails it, so it must be flagged invalid.
PBT_ADD_BUGGY = (
    "from hypothesis import given\n"
    "from hypothesis import strategies as st\n\n"
    "def test_pbt(fn):\n"
    "    @given(a=st.integers(), b=st.integers())\n"
    "    def _t(a, b):\n"
    "        assert fn(a, b) > 0\n"
    "    _t()\n"
)

REFERENCE_ADD = "def add(a, b):\n    return a + b"


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
        "INSERT OR IGNORE INTO tasks(task_id, dataset, prompt, reference_solution, "
        "entry_point, setup, prelude, per_timeout, io_mode, difficulty) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (t.task_id, t.dataset, t.prompt, t.reference_solution, t.entry_point,
         t.setup, t.prelude, t.per_timeout, t.io_mode, t.difficulty),
    )


def _insert_suite(conn: sqlite3.Connection, s: Suite) -> None:
    """Insert a Suite row, ignoring an existing one with the same id."""
    conn.execute(
        "INSERT OR IGNORE INTO suites(suite_id, task_id, suite_model, kind, code, valid) "
        "VALUES (?,?,?,?,?,?)",
        (s.suite_id, s.task_id, s.suite_model, s.kind, s.code, s.valid),
    )


def _valid_of(conn: sqlite3.Connection, suite_id: str):
    """Return the stored `valid` value for a suite (None, 0, or 1)."""
    return conn.execute("SELECT valid FROM suites WHERE suite_id = ?", (suite_id,)).fetchone()["valid"]


def test_reference_passing_pbt_is_marked_valid():
    """A correct pbt the reference solution passes earns valid = 1."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="add",
                                reference_solution=REFERENCE_ADD, entry_point="add"))
        suite = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_ADD_CORRECT)
        _insert_suite(conn, suite)
        conn.commit()
        checked = validity.validate_pbts(conn)
        assert checked == 1, "the one pbt with a reference solution must be checked"
        assert _valid_of(conn, suite.suite_id) == 1, "reference passes the correct pbt: valid=1"


def test_reference_failing_pbt_is_marked_invalid():
    """A buggy pbt the reference solution fails earns valid = 0."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="add",
                                reference_solution=REFERENCE_ADD, entry_point="add"))
        suite = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_ADD_BUGGY)
        _insert_suite(conn, suite)
        conn.commit()
        checked = validity.validate_pbts(conn)
        assert checked == 1, "the buggy pbt still has a reference solution and is checked"
        assert _valid_of(conn, suite.suite_id) == 0, "reference fails the buggy pbt: valid=0"


def test_missing_reference_is_skipped_and_left_null():
    """A task with no reference_solution is skipped; its pbt stays valid = NULL."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="add",
                                reference_solution="", entry_point="add"))
        suite = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_ADD_CORRECT)
        _insert_suite(conn, suite)
        conn.commit()
        checked = validity.validate_pbts(conn)
        assert checked == 0, "a pbt with no reference solution is not checked"
        assert _valid_of(conn, suite.suite_id) is None, "skipped pbt must stay unchecked (NULL)"


def test_dataset_filter_restricts_checked_pbts():
    """The dataset filter checks only pbts whose task is in that dataset."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="add",
                                reference_solution=REFERENCE_ADD, entry_point="add"))
        _insert_task(conn, Task(task_id="t2", dataset="humaneval", prompt="add",
                                reference_solution=REFERENCE_ADD, entry_point="add"))
        in_set = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_ADD_CORRECT)
        out_set = Suite(task_id="t2", suite_model="gpt-4", kind="pbt", code=PBT_ADD_CORRECT)
        _insert_suite(conn, in_set)
        _insert_suite(conn, out_set)
        conn.commit()
        checked = validity.validate_pbts(conn, dataset="mbppplus")
        assert checked == 1, "only the mbppplus pbt is checked"
        assert _valid_of(conn, in_set.suite_id) == 1, "the in-dataset pbt is verified valid"
        assert _valid_of(conn, out_set.suite_id) is None, "the other dataset is left untouched"


def test_validate_pbts_is_idempotent():
    """Re-running the filter recomputes the same verdict for the same inputs."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="add",
                                reference_solution=REFERENCE_ADD, entry_point="add"))
        suite = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_ADD_CORRECT)
        _insert_suite(conn, suite)
        conn.commit()
        first = validity.validate_pbts(conn)
        second = validity.validate_pbts(conn)
        assert first == second == 1, "both runs check the same one pbt"
        assert _valid_of(conn, suite.suite_id) == 1, "the verdict is stable across runs"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
