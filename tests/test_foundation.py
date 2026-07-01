"""Foundation smoke tests: schema, content-hash IDs, dedup, idempotency, read-only guard.

No API calls, no scoring; just the storage layer and types. Runs under pytest or standalone:

    python -m pytest tests/test_foundation.py -q
    python tests/test_foundation.py
"""
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbt import db, ids
from pbt.core import Program, Suite, Task


@contextmanager
def temp_db():
    """Yield a read-write connection to a throwaway database.

    Creates the schema in a temporary directory and tears it down on exit, so each test runs against an isolated store. Deliberately a context manager rather than a pytest fixture so the file still runs standalone (see __main__).

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
        "INSERT OR IGNORE INTO suites(suite_id, task_id, suite_model, kind, code) VALUES (?,?,?,?,?)",
        (s.suite_id, s.task_id, s.suite_model, s.kind, s.code),
    )


def test_ids_are_deterministic_and_model_sensitive():
    """Same (task, model, code) hashes alike; a different model gives a distinct id."""
    a = ids.program_id("t1", "gpt-4", "def f(): return 1")
    b = ids.program_id("t1", "gpt-4", "def f(): return 1")
    c = ids.program_id("t1", "claude", "def f(): return 1")  # different model
    assert a == b, "same (task, model, code) must hash identically"
    assert a != c, "different model must produce a distinct id (provenance preserved)"


def test_types_compute_their_own_ids():
    """A constructed Program/Suite carries the same id the ids module would compute."""
    p = Program(task_id="t1", prog_model="gpt-4", code="x = 1")
    assert p.program_id == ids.program_id("t1", "gpt-4", "x = 1")
    s = Suite(task_id="t1", suite_model="claude", kind="pbt", code="# pbt")
    assert s.suite_id == ids.suite_id("t1", "claude", "# pbt")


def test_round_trip_and_dedup():
    """Inserting identical program code three times collapses to one row."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="add two numbers",
                                reference_solution="def add(a,b): return a+b"))
        p = Program(task_id="t1", prog_model="gpt-4", code="def add(a,b): return a+b")
        for _ in range(3):  # the content-hash PK should reject the duplicates
            _insert_program(conn, p)
        conn.commit()
        (n,) = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        assert n == 1, "content-hash PK must dedup identical programs"


def test_generations_capture_multiplicity_while_programs_dedup():
    """Five draws of identical code give five generation events but one program."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="p"))
        p = Program(task_id="t1", prog_model="gpt-4", code="def add(a,b): return a+b")
        _insert_program(conn, p)
        for i in range(5):  # same program produced on every draw
            conn.execute(
                "INSERT OR IGNORE INTO generations(gen_id, task_id, prog_model, sample_index, "
                "program_id) VALUES (?,?,?,?,?)",
                (ids.gen_id("t1", "gpt-4", i), "t1", "gpt-4", i, p.program_id),
            )
        conn.commit()
        (progs,) = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        (mult,) = conn.execute("SELECT COUNT(*) FROM generations WHERE program_id=?",
                               (p.program_id,)).fetchone()
        assert progs == 1 and mult == 5, "1 deduped program, 5 generation events"


def test_results_pk_is_idempotent_cache():
    """A second score for the same (program, suite) no-ops; the first result wins."""
    with temp_db() as conn:
        _insert_task(conn, Task(task_id="t1", dataset="mbppplus", prompt="p"))
        p = Program(task_id="t1", prog_model="gpt-4", code="c")
        s = Suite(task_id="t1", suite_model="claude", kind="pbt", code="pbt")
        _insert_program(conn, p)
        _insert_suite(conn, s)
        passing = (p.program_id, s.suite_id, "t1", "gpt-4", "claude", "pbt", 1, 1.0)
        failing = (p.program_id, s.suite_id, "t1", "gpt-4", "claude", "pbt", 0, 0.0)
        conn.execute("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?,?,?)", passing)
        conn.execute("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?,?,?)", failing)
        conn.commit()
        rows = conn.execute("SELECT passed FROM results").fetchall()
        assert len(rows) == 1 and rows[0]["passed"] == 1, "first result wins; re-score no-ops"


def test_read_only_connection_blocks_writes():
    """A read-only analysis connection raises rather than mutating the store."""
    with tempfile.TemporaryDirectory() as d:  # path must outlive the rw connection
        path = os.path.join(d, "pbt.db")
        db.connect(path).close()  # create the file and schema
        ro = db.connect_ro(path)
        try:
            ro.execute("INSERT INTO tasks(task_id, dataset, prompt) VALUES ('x','y','z')")
            raised = False
        except sqlite3.OperationalError:
            raised = True
        assert raised, "analysis connection must be physically unable to write"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
