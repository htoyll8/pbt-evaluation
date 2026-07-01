"""Orchestrate the cross product: score every program against every suite.

The study's one measurement is score(program, suite); this module drives it over
the whole store. For each task it loads that task's programs and suites and scores
the full within-task cross product (every program against every unit, private, and
pbt suite), filling the `results` table. Scoring stays within a task because a
program and a suite only share grading context (setup, prelude, io_mode) when they
belong to the same task.

Idempotency:
    score() is cache-first on the `results` primary key (program_id, suite_id), so
    re-running evaluate re-grades nothing already stored. A fast pre-check skips
    pairs already present before calling score(), so a resumed run does no
    redundant work.

Reconstructing dataclasses from rows:
    Task, Program, and Suite are frozen dataclasses whose content-hash IDs are
    recomputed at construction. We rebuild each object from its natural identity
    fields (for a Program: task_id, prog_model, code) and let the dataclass
    recompute the id, then assert the recomputed id equals the id stored on the
    row. That turns every load into a content-integrity check: a mismatch means the
    stored bytes drifted from their hash, which must never happen in an append-only
    store, so we fail loudly rather than score a mislabeled artifact.
"""
import sqlite3

from pbt.core import Program, Suite, Task, score


def _load_tasks(conn: sqlite3.Connection, dataset: str | None) -> list[Task]:
    """Load tasks from the store, optionally filtered to one dataset.

    Args:
        conn: Open connection to the store.
        dataset: If given, restrict to tasks from this dataset; otherwise load all.

    Returns:
        The reconstructed Task objects, ordered by task_id for a stable run.
    """
    if dataset is None:
        rows = conn.execute(
            "SELECT task_id, dataset, prompt, reference_solution, entry_point, setup, "
            "prelude, per_timeout, io_mode, difficulty FROM tasks ORDER BY task_id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT task_id, dataset, prompt, reference_solution, entry_point, setup, "
            "prelude, per_timeout, io_mode, difficulty FROM tasks WHERE dataset = ? "
            "ORDER BY task_id",
            (dataset,),
        ).fetchall()
    return [
        Task(
            task_id=row["task_id"],
            dataset=row["dataset"],
            prompt=row["prompt"],
            reference_solution=row["reference_solution"] or "",
            entry_point=row["entry_point"] or "",
            setup=row["setup"],
            prelude=row["prelude"],
            per_timeout=row["per_timeout"],
            io_mode=row["io_mode"],
            difficulty=row["difficulty"],
        )
        for row in rows
    ]


def _load_programs(conn: sqlite3.Connection, task_id: str) -> list[Program]:
    """Load a task's programs, recomputing and verifying each content-hash id.

    Args:
        conn: Open connection to the store.
        task_id: ID of the task whose programs to load.

    Returns:
        The reconstructed Program objects for the task, ordered by program_id.

    Raises:
        ValueError: If a row's stored program_id does not match the id recomputed
            from its (task_id, prog_model, code) fields.
    """
    rows = conn.execute(
        "SELECT program_id, task_id, prog_model, code FROM programs "
        "WHERE task_id = ? ORDER BY program_id",
        (task_id,),
    ).fetchall()
    programs = []
    for row in rows:
        program = Program(
            task_id=row["task_id"],
            prog_model=row["prog_model"],
            code=row["code"],
        )
        if program.program_id != row["program_id"]:
            raise ValueError(
                f"program_id mismatch: stored {row['program_id']!r} but content "
                f"hashes to {program.program_id!r}"
            )
        programs.append(program)
    return programs


def _load_suites(conn: sqlite3.Connection, task_id: str) -> list[Suite]:
    """Load a task's suites, recomputing and verifying each content-hash id.

    Args:
        conn: Open connection to the store.
        task_id: ID of the task whose suites to load.

    Returns:
        The reconstructed Suite objects for the task (every kind), ordered by
        suite_id.

    Raises:
        ValueError: If a row's stored suite_id does not match the id recomputed
            from its (task_id, suite_model, code) fields.
    """
    rows = conn.execute(
        "SELECT suite_id, task_id, suite_model, kind, code, valid FROM suites "
        "WHERE task_id = ? ORDER BY suite_id",
        (task_id,),
    ).fetchall()
    suites = []
    for row in rows:
        valid = None if row["valid"] is None else bool(row["valid"])
        suite = Suite(
            task_id=row["task_id"],
            suite_model=row["suite_model"],
            kind=row["kind"],
            code=row["code"],
            valid=valid,
        )
        if suite.suite_id != row["suite_id"]:
            raise ValueError(
                f"suite_id mismatch: stored {row['suite_id']!r} but content "
                f"hashes to {suite.suite_id!r}"
            )
        suites.append(suite)
    return suites


def _is_scored(conn: sqlite3.Connection, program_id: str, suite_id: str) -> bool:
    """Report whether a (program, suite) pair already has a stored result.

    A fast pre-check so a resumed run skips work score() would no-op anyway.

    Args:
        conn: Open connection to the store.
        program_id: ID of the candidate program.
        suite_id: ID of the suite it would be scored against.

    Returns:
        True if a results row already exists for the pair, otherwise False.
    """
    row = conn.execute(
        "SELECT 1 FROM results WHERE program_id = ? AND suite_id = ?",
        (program_id, suite_id),
    ).fetchone()
    return row is not None


def evaluate(conn: sqlite3.Connection, dataset: str | None = None) -> int:
    """Score the program-by-suite cross product for every task in the store.

    Loads each task (optionally filtered to one dataset), then for that task scores
    every program against every suite (unit, private, and pbt) by calling score(),
    which writes the verdict to `results`. Idempotent: pairs already in `results`
    are skipped, and score() is itself cache-first, so re-running scores nothing new.

    Args:
        conn: Open read-write connection to the store.
        dataset: If given, restrict scoring to tasks from this dataset; otherwise
            evaluate every task.

    Returns:
        The number of (program, suite) pairs scored on this call, counting only
        pairs not already present in `results`.
    """
    scored = 0
    for task in _load_tasks(conn, dataset):
        programs = _load_programs(conn, task.task_id)
        suites = _load_suites(conn, task.task_id)
        pairs = 0
        for program in programs:
            for suite in suites:
                if _is_scored(conn, program.program_id, suite.suite_id):
                    continue
                score(conn, program, suite, task)
                pairs += 1
        scored += pairs
        print(f"task {task.task_id}: {pairs} pairs "
              f"({len(programs)} programs x {len(suites)} suites)")
    return scored
