"""Migrate a store to the weak-gated schema: unit = original suite, private = augmented.

Why this exists
---------------
Before this change the MBPP+ loader seeded the EvalPlus *plus* harness as the task's
kind="unit" suite. Since pbt.analyze gates eligibility on "program passed its unit suite",
MBPP+ rates were computed over only those programs that already survive the augmented
harness, silently dropping every program that passes the 3 base asserts but fails plus.
That dropped set is the overfit population the study is about. HumanEval was unaffected
(its unit suite was already the weak check() asserts), so the two benchmarks were reporting
different quantities under one name.

The loaders now seed the original suite as kind="unit" and the augmented one as
kind="private". This script rebuilds an existing store against those loaders without
re-paying for generation.

What is reused vs recomputed
----------------------------
Reused (copied verbatim, ids are content-hashes over unchanged inputs):
  programs, generations  -- program_id = hash(task_id, prog_model, code); nothing changed.
  suites kind='pbt'      -- suite_id = hash(task_id, suite_model, code); nothing changed.
                            Their `valid` flag is copied too: validity is the reference
                            solution's verdict on the PBT, and neither moved.
  results kind='pbt'     -- keyed (program_id, suite_id), both unchanged. This is the
                            expensive half (each is a 100-example Hypothesis subprocess),
                            so copying it is the whole point of migrating rather than
                            re-running from scratch.

Recomputed (cheap, local, no API):
  results kind='unit'    -- the unit suite is now different tests, so old rows do not apply.
  results kind='private' -- new suite, never scored before.

Usage: python scripts/migrate_weak_gate.py <old.db> <new.db> --dataset mbppplus --n-tasks 378
"""
import argparse
import sqlite3

from pbt import analyze, db, evaluate
from pbt.seed import seed_dataset

_COPY_TABLES = (
    ("programs", "SELECT * FROM old.programs"),
    ("generations", "SELECT * FROM old.generations"),
    ("suites", "SELECT * FROM old.suites WHERE kind = 'pbt'"),
    ("results", "SELECT * FROM old.results WHERE kind = 'pbt'"),
)


def _cols(conn: sqlite3.Connection, table: str) -> str:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return ", ".join(r[1] for r in rows)


def migrate(old_db: str, new_db: str, dataset: str, n_tasks: int) -> None:
    conn = db.connect(new_db)

    n_seeded = seed_dataset(conn, dataset, n_tasks)
    unit_n = conn.execute("SELECT COUNT(*) FROM suites WHERE kind='unit'").fetchone()[0]
    priv_n = conn.execute("SELECT COUNT(*) FROM suites WHERE kind='private'").fetchone()[0]
    print(f"seeded {n_seeded} tasks -> {unit_n} unit suites, {priv_n} private suites")
    if priv_n == 0:
        raise SystemExit("no private suites seeded; loader did not supply private_tests")

    conn.execute("ATTACH DATABASE ? AS old", (old_db,))
    for table, select in _COPY_TABLES:
        cols = _cols(conn, table)
        # INSERT OR IGNORE: the store is append-only and keyed by content hash, so a row
        # already present is by definition identical. Makes the migration re-runnable.
        conn.execute(f"INSERT OR IGNORE INTO {table}({cols}) {select.replace('*', cols, 1)}")
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"copied {table:12s} -> {n} rows now present")
    conn.commit()
    conn.execute("DETACH DATABASE old")

    carried = conn.execute(
        "SELECT COUNT(*) FROM results WHERE kind='pbt'").fetchone()[0]
    print(f"carried {carried} pbt verdicts (no Hypothesis re-runs needed)")

    print("scoring unit + private suites (local, no API)...")
    n_pairs = evaluate.evaluate(conn, dataset)
    print(f"evaluate touched {n_pairs} pairs")

    conn.close()
    analyze.print_report(db.connect_ro(new_db))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("old_db")
    ap.add_argument("new_db")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-tasks", type=int, required=True)
    a = ap.parse_args()
    migrate(a.old_db, a.new_db, a.dataset, a.n_tasks)
