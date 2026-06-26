"""Tests for the overfit-catch metric in pbt.analyze.

Builds a tiny store by inserting synthetic `results` rows directly (no real
scoring), then checks the overall rate and the cross-model matrix. Runs under
pytest or standalone:

    python -m pytest tests/test_analyze.py -q
    python tests/test_analyze.py
"""
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbt import analyze, db


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
        # Insert synthetic results rows directly without seeding parent tables.
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            yield conn
        finally:
            conn.close()


def _result(conn: sqlite3.Connection, program_id: str, suite_id: str, *,
            prog_model: str, suite_model: str, kind: str, passed: int) -> None:
    """Insert one synthetic results row (pass_fraction mirrors passed)."""
    conn.execute(
        "INSERT OR IGNORE INTO results(program_id, suite_id, task_id, prog_model, "
        "suite_model, kind, passed, pass_fraction) VALUES (?,?,?,?,?,?,?,?)",
        (program_id, suite_id, "t1", prog_model, suite_model, kind, passed, float(passed)),
    )


def _seed_three_programs(conn: sqlite3.Connection) -> None:
    """Seed a store with one overfit catch, one clean pass, one ineligible program.

    All programs are written by model "A"; all pbts by model "B".
        p_overfit: passes unit, fails pbt   -> eligible and caught.
        p_clean:   passes unit, passes pbt  -> eligible, not caught.
        p_bad:     fails unit, fails pbt    -> ineligible (excluded entirely).
    """
    _result(conn, "p_overfit", "s_unit", prog_model="A", suite_model="A", kind="unit", passed=1)
    _result(conn, "p_overfit", "s_pbt", prog_model="A", suite_model="B", kind="pbt", passed=0)

    _result(conn, "p_clean", "s_unit", prog_model="A", suite_model="A", kind="unit", passed=1)
    _result(conn, "p_clean", "s_pbt", prog_model="A", suite_model="B", kind="pbt", passed=1)

    _result(conn, "p_bad", "s_unit", prog_model="A", suite_model="A", kind="unit", passed=0)
    _result(conn, "p_bad", "s_pbt", prog_model="A", suite_model="B", kind="pbt", passed=0)
    conn.commit()


def test_overall_rate_counts_only_unit_passers():
    """Two eligible pairs, one caught: the rate is 0.5 and the unit-failer is excluded."""
    with temp_db() as conn:
        _seed_three_programs(conn)
        out = analyze.overfit_catch_rate(conn)
        assert out["eligible"] == 2, "only the two unit-passing programs are eligible"
        assert out["caught"] == 1, "only p_overfit fails its pbt"
        assert out["rate"] == 0.5, f"expected 0.5 catch rate, got {out['rate']}"


def test_matrix_is_keyed_by_prog_and_suite_model():
    """The single A-program/B-pbt cell carries the same eligible, caught, and rate."""
    with temp_db() as conn:
        _seed_three_programs(conn)
        matrix = analyze.overfit_matrix(conn)
        assert len(matrix) == 1, "all eligible pairs fall in one (A, B) cell"
        cell = matrix[0]
        assert cell["prog_model"] == "A" and cell["suite_model"] == "B", \
            "cell must be keyed by (prog_model=A, suite_model=B)"
        assert cell["eligible"] == 2 and cell["caught"] == 1, "2 eligible, 1 caught"
        assert cell["rate"] == 0.5, f"expected 0.5 in the cell, got {cell['rate']}"


def test_matrix_separates_distinct_model_pairs():
    """Pbts from two suite models split into two cells with independent rates."""
    with temp_db() as conn:
        # Program by A passes unit; pbt from B catches it, pbt from C does not.
        _result(conn, "p1", "s_unit", prog_model="A", suite_model="A", kind="unit", passed=1)
        _result(conn, "p1", "s_pbt_b", prog_model="A", suite_model="B", kind="pbt", passed=0)
        _result(conn, "p1", "s_pbt_c", prog_model="A", suite_model="C", kind="pbt", passed=1)
        conn.commit()
        by_pair = {(c["prog_model"], c["suite_model"]): c for c in analyze.overfit_matrix(conn)}
        assert by_pair[("A", "B")]["rate"] == 1.0, "B's pbt catches A's overfit"
        assert by_pair[("A", "C")]["rate"] == 0.0, "C's pbt misses it"


def test_empty_db_is_safe_zero_and_none():
    """An empty store yields zero counts and a None rate, never a divide-by-zero."""
    with temp_db() as conn:
        out = analyze.overfit_catch_rate(conn)
        assert out["eligible"] == 0 and out["caught"] == 0, "no rows means zero counts"
        assert out["rate"] is None, "rate must be None, not a ZeroDivisionError"
        assert analyze.overfit_matrix(conn) == [], "empty store yields an empty matrix"


def test_analysis_runs_on_read_only_connection():
    """The metric computes over a read-only connection (analysis never writes)."""
    with tempfile.TemporaryDirectory() as d:  # path must outlive the rw connection
        path = os.path.join(d, "pbt.db")
        rw = db.connect(path)
        rw.execute("PRAGMA foreign_keys = OFF")  # seed results without parent rows
        _seed_three_programs(rw)
        rw.close()
        ro = db.connect_ro(path)
        try:
            out = analyze.overfit_catch_rate(ro)
            assert out["rate"] == 0.5, "read-only analysis reproduces the 0.5 rate"
        finally:
            ro.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
