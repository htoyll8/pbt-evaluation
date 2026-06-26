"""Scoring tests: the cache, the unit/private worker path, and the PBT path.

Exercises pbt.core.score end to end against a throwaway store. Runs under pytest
or standalone:

    python -m pytest tests/test_score.py -q
    python tests/test_score.py
"""
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbt import db
from pbt.core import Program, Suite, Task, score


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


def _unit_blob(tests: list, io_mode: str = "function") -> str:
    """Serialize a unit/private suite blob exactly as pbt.seed does."""
    return json.dumps({"io_mode": io_mode, "tests": tests}, sort_keys=True)


# A Hypothesis property that the candidate `double` must satisfy for every integer.
PBT_DOUBLE = (
    "from hypothesis import given\n"
    "from hypothesis import strategies as st\n\n"
    "def test_pbt(fn):\n"
    "    @given(x=st.integers())\n"
    "    def _t(x):\n"
    "        assert fn(x) == 2 * x\n"
    "    _t()\n"
)

# A Hypothesis property that the candidate `add` must satisfy for every integer pair.
PBT_ADD = (
    "from hypothesis import given\n"
    "from hypothesis import strategies as st\n\n"
    "def test_pbt(fn):\n"
    "    @given(a=st.integers(), b=st.integers())\n"
    "    def _t(a, b):\n"
    "        assert fn(a, b) == a + b\n"
    "    _t()\n"
)


def test_cache_hit_returns_without_recompute():
    """A pre-stored result is returned verbatim, so scoring never recomputes it."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="add"))
        # A program that would actually score 0 if run against this suite...
        prog = Program(task_id="t1", prog_model="gpt-4", code="def add(a, b):\n    return 999")
        suite = Suite(task_id="t1", suite_model="benchmark", kind="unit",
                      code=_unit_blob(["assert add(1, 2) == 3"]))
        _insert_program(conn, prog)
        _insert_suite(conn, suite)
        # ...but the cache says it passed; score() must trust the cache, not recompute.
        conn.execute(
            "INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?,?,?)",
            (prog.program_id, suite.suite_id, "t1", "gpt-4", "benchmark", "unit", 1, 1.0),
        )
        conn.commit()
        result = score(conn, prog, suite, Task(task_id="t1", dataset="mbppplus", prompt="add"))
        assert result.passed is True and result.pass_fraction == 1.0, \
            "cache hit must return the stored verdict without recomputing"


def test_passing_unit_suite():
    """A correct program passes its benchmark unit suite (pass_fraction 1.0)."""
    with temp_db() as conn:
        task = Task(task_id="t1", dataset="mbppplus", prompt="add two numbers")
        _insert_task(conn, task)
        prog = Program(task_id="t1", prog_model="gpt-4", code="def add(a, b):\n    return a + b")
        suite = Suite(task_id="t1", suite_model="benchmark", kind="unit",
                      code=_unit_blob(["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]))
        _insert_program(conn, prog)
        _insert_suite(conn, suite)
        conn.commit()
        result = score(conn, prog, suite, task)
        assert result.passed and result.pass_fraction == 1.0, \
            "a correct program must pass every unit case"
        (n,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        assert n == 1, "the fresh verdict must be persisted exactly once"


def test_pbt_catches_wrong_program():
    """A PBT finds a counterexample for a deliberately wrong program (passed False)."""
    with temp_db() as conn:
        task = Task(task_id="t1", dataset="mbppplus", prompt="double a number")
        _insert_task(conn, task)
        wrong = Program(task_id="t1", prog_model="gpt-4",
                        code="def double(x):\n    return x + 1")  # only correct at x == 1
        suite = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_DOUBLE)
        _insert_program(conn, wrong)
        _insert_suite(conn, suite)
        conn.commit()
        result = score(conn, wrong, suite, task)
        assert result.passed is False and result.pass_fraction == 0.0, \
            "the PBT must catch the wrong program with a counterexample"


def test_pbt_passes_correct_program():
    """A correct program survives the PBT with no counterexample (passed True)."""
    with temp_db() as conn:
        task = Task(task_id="t1", dataset="mbppplus", prompt="double a number")
        _insert_task(conn, task)
        right = Program(task_id="t1", prog_model="gpt-4",
                        code="def double(x):\n    return 2 * x")
        suite = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_DOUBLE)
        _insert_program(conn, right)
        _insert_suite(conn, suite)
        conn.commit()
        result = score(conn, right, suite, task)
        assert result.passed is True and result.pass_fraction == 1.0, \
            "a correct program must pass the PBT (no counterexample)"


def test_pbt_binds_entry_point_over_preceding_helper():
    """entry_point binds the named function even when a helper `def` precedes it."""
    with temp_db() as conn:
        # entry_point="add" must be bound, not the earlier top-level def `helper`.
        task = Task(task_id="t1", dataset="mbppplus", prompt="add two numbers",
                    entry_point="add")
        _insert_task(conn, task)
        prog = Program(task_id="t1", prog_model="gpt-4",
                       code="def helper(x):\n    return x * 2\n\ndef add(a, b):\n    return a + b")
        suite = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_ADD)
        _insert_program(conn, prog)
        _insert_suite(conn, suite)
        conn.commit()
        result = score(conn, prog, suite, task)
        assert result.passed is True and result.pass_fraction == 1.0, \
            "entry_point must bind `add`, not the preceding helper `def`"


def test_pbt_falls_back_to_first_def_without_entry_point():
    """With no entry_point set, binding falls back to the first top-level `def`."""
    with temp_db() as conn:
        task = Task(task_id="t1", dataset="mbppplus", prompt="double a number")  # entry_point=""
        _insert_task(conn, task)
        prog = Program(task_id="t1", prog_model="gpt-4",
                       code="def double(x):\n    return 2 * x")
        suite = Suite(task_id="t1", suite_model="claude", kind="pbt", code=PBT_DOUBLE)
        _insert_program(conn, prog)
        _insert_suite(conn, suite)
        conn.commit()
        result = score(conn, prog, suite, task)
        assert result.passed is True and result.pass_fraction == 1.0, \
            "the first top-level def must be bound when no entry_point is given"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
