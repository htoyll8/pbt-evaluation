"""The PBT validity filter: does the reference solution pass each property?

A PBT only earns its place in the overfit metric if it is itself correct. A buggy
PBT (one that asserts a too-strong property or violates the task's preconditions)
would fail even the task's canonical solution, so any program it "catches" is a
false positive. This module runs each pbt suite against its task's reference
solution and records the verdict on the suite's `valid` column: 1 when the
reference passes (the PBT is valid), 0 when it fails (the PBT is buggy).

Append-only exception:
    The store is otherwise append-only (facts and content are written once with
    INSERT OR IGNORE, never overwritten). `valid` is the single sanctioned
    exception: it is not a fact but a derived verdict about a suite, recomputed
    from the suite's own code and its task's reference solution. UPDATE-ing it is
    therefore allowed here and nowhere else, and it is idempotent: re-running this
    filter recomputes the same verdict for the same inputs.
"""
import sqlite3

from pbt.core import run_property


def validate_pbts(conn: sqlite3.Connection, dataset: str | None = None) -> int:
    """Check each pbt suite against its task's reference solution.

    For every `kind="pbt"` suite (optionally restricted to one dataset), runs the
    suite's property against the task's reference solution and writes the verdict
    to the suite's `valid` column: 1 if the reference passed, 0 if it failed. A
    suite whose task has no reference solution is skipped with a warning and left
    unchecked (valid stays NULL). Idempotent: re-running recomputes the same
    verdicts.

    Args:
        conn: Open read-write connection to the store.
        dataset: If given, only check pbt suites whose task belongs to this
            dataset; otherwise check every pbt suite.

    Returns:
        The number of pbt suites checked (those with a non-empty reference
        solution); skipped suites are not counted.
    """
    query = (
        "SELECT s.suite_id AS suite_id, s.code AS code, "
        "t.reference_solution AS reference_solution, t.entry_point AS entry_point, "
        "t.per_timeout AS per_timeout "
        "FROM suites AS s JOIN tasks AS t ON t.task_id = s.task_id "
        "WHERE s.kind = 'pbt'"
    )
    params: tuple = ()
    if dataset is not None:
        query += " AND t.dataset = ?"
        params = (dataset,)

    rows = conn.execute(query, params).fetchall()
    checked = 0
    for row in rows:
        reference = row["reference_solution"] or ""
        if not reference.strip():
            print(f"warning: skipping suite {row['suite_id']}: task has no "
                  "reference_solution (leaving valid NULL)")
            continue
        passed = run_property(
            reference, row["code"], row["entry_point"] or "", row["per_timeout"]
        )
        conn.execute(
            "UPDATE suites SET valid = ? WHERE suite_id = ?",
            (1 if passed else 0, row["suite_id"]),
        )
        checked += 1
    conn.commit()
    return checked
