"""Load existing pbt_data artifacts into the store, no model APIs required.

The generators in this repo wrote two families of JSONL files that this module
replays into the normalized store so a pilot can run offline:

PBT files (pbt_data/pbts_*.jsonl):
    One JSON object per line authored by a model: {task_id, dataset, pbt_code,
    pbt_valid, generation_model, ...}. Each becomes one kind="pbt" Suite whose
    `code` is the raw Hypothesis module defining `test_pbt(fn)`, stored verbatim
    (not a JSON blob) because pbt.core.score execs it directly. `pbt_valid` maps
    onto the suite's validity flag and `generation_model` onto its suite_model.

Program files (results/*.jsonl, results/seeds_*.jsonl):
    One JSON object per task in the single-attempt trajectory format that
    scripts/generate_seeds.py emits: {task_id, dataset, trajectories: [{seed_index,
    initial_program, ...}], generation_model, ...}. Each trajectory's
    initial_program becomes one deduped Program plus one generations row keyed by
    its seed_index, so identical code sampled repeatedly stays one program with
    many generation events.

Task alignment:
    The grading loaders (and therefore pbt.seed) key tasks by str(ex["task_id"]),
    so an MBPP+ task_id is the string "2", not the integer 2 that the JSONL holds.
    Every import coerces task_id with str() to match the seeded rows; otherwise the
    foreign key would point at a task that does not exist. Imports referencing a
    task that has not been seeded are skipped with a warning, so seeding must run
    first for those rows to land.

Every write is INSERT OR IGNORE on a content-hash id, so re-importing a file adds
nothing new.
"""
import json
import sqlite3
import sys

from pbt import ids
from pbt.core import Program, Suite


def _warn(message: str) -> None:
    """Print a warning to stderr.

    Args:
        message: The text to emit, prefixed so import noise is easy to filter.
    """
    print(f"[importer] {message}", file=sys.stderr)


def _task_exists(conn: sqlite3.Connection, task_id: str) -> bool:
    """Report whether a task row is present, to honor the foreign key.

    Args:
        conn: Open connection to the store.
        task_id: Candidate task id, already coerced to str.

    Returns:
        True if a tasks row with this id exists, False otherwise.
    """
    row = conn.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    return row is not None


def _read_jsonl(path: str):
    """Yield parsed JSON objects from a JSONL file, skipping blank lines.

    Args:
        path: Filesystem path to a newline-delimited JSON file.

    Yields:
        One decoded object per non-empty line.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _insert_suite(conn: sqlite3.Connection, suite: Suite) -> None:
    """Insert a suite row, ignoring an existing row with the same id.

    Args:
        conn: Open read-write connection to the store.
        suite: The deduped suite to persist.
    """
    conn.execute(
        "INSERT OR IGNORE INTO suites(suite_id, task_id, suite_model, kind, code, valid) "
        "VALUES (?,?,?,?,?,?)",
        (suite.suite_id, suite.task_id, suite.suite_model, suite.kind, suite.code,
         suite.valid),
    )


def _insert_program(conn: sqlite3.Connection, program: Program) -> None:
    """Insert a program row, ignoring an existing row with the same id.

    Args:
        conn: Open read-write connection to the store.
        program: The deduped candidate program to persist.
    """
    conn.execute(
        "INSERT OR IGNORE INTO programs(program_id, task_id, prog_model, code) "
        "VALUES (?,?,?,?)",
        (program.program_id, program.task_id, program.prog_model, program.code),
    )


def _insert_generation(conn: sqlite3.Connection, task_id: str, prog_model: str,
                       sample_index: int, program_id: str) -> None:
    """Insert one generation event, ignoring a duplicate (task, model, index).

    Args:
        conn: Open read-write connection to the store.
        task_id: Id of the sampled task.
        prog_model: Model that produced the sample.
        sample_index: Zero or one based index of this draw within the task/model.
        program_id: Id of the deduped program this draw produced.
    """
    conn.execute(
        "INSERT OR IGNORE INTO generations(gen_id, task_id, prog_model, sample_index, "
        "program_id) VALUES (?,?,?,?,?)",
        (ids.gen_id(task_id, prog_model, sample_index), task_id, prog_model,
         sample_index, program_id),
    )


def import_pbts(conn: sqlite3.Connection, path: str) -> int:
    """Import a PBTs JSONL file as kind="pbt" suites.

    Each record's `pbt_code` is stored verbatim as the suite code (the raw
    Hypothesis module defining `test_pbt(fn)` that pbt.core.score execs), authored
    by the record's `generation_model` and flagged with its `pbt_valid` verdict.
    Records whose task has not been seeded are skipped with a warning to respect
    the suites foreign key.

    Args:
        conn: Open read-write connection to the store.
        path: Path to a pbt_data/pbts_*.jsonl file.

    Returns:
        The number of suites written or already present (records processed minus
        those skipped for a missing task, code, or model).
    """
    written = 0
    for record in _read_jsonl(path):
        task_id = str(record["task_id"])
        code = record.get("pbt_code")
        model = record.get("generation_model")
        if not code or not model:
            _warn(f"skipping pbt for task {task_id!r}: missing pbt_code or generation_model")
            continue
        if not _task_exists(conn, task_id):
            _warn(f"skipping pbt for task {task_id!r}: task not seeded (run seeding first)")
            continue
        suite = Suite(
            task_id=task_id,
            suite_model=model,
            kind="pbt",
            code=code,
            valid=record.get("pbt_valid"),
        )
        _insert_suite(conn, suite)
        written += 1
    conn.commit()
    return written


def import_programs(conn: sqlite3.Connection, path: str, prog_model: str | None = None) -> int:
    """Import candidate programs from a trajectory JSONL file.

    Reads the single-attempt trajectory format that scripts/generate_seeds.py and
    the self-correction runs emit: one record per task carrying a `trajectories`
    list whose entries hold an `initial_program`. Each program becomes one deduped
    Program plus one generations row keyed by the trajectory's `seed_index`.
    Records whose task has not been seeded are skipped with a warning to respect
    the programs foreign key.

    Args:
        conn: Open read-write connection to the store.
        path: Path to a trajectory JSONL file (e.g. results/seeds_*.jsonl).
        prog_model: Model that authored the programs, overriding any
            `generation_model` or `prog_model` carried in the file. Required when
            the file does not name its model.

    Returns:
        The number of programs written or already present (trajectories processed
        minus those skipped for a missing task, code, or model).

    Raises:
        ValueError: If a record names no model and no prog_model override is given.
    """
    written = 0
    for record in _read_jsonl(path):
        task_id = str(record["task_id"])
        model = prog_model or record.get("generation_model") or record.get("prog_model")
        if not model:
            raise ValueError(
                f"no model for task {task_id!r}: pass prog_model since the file does not "
                f"carry generation_model"
            )
        if not _task_exists(conn, task_id):
            _warn(f"skipping programs for task {task_id!r}: task not seeded (run seeding first)")
            continue
        for trajectory in record.get("trajectories", []):
            code = trajectory.get("initial_program") or trajectory.get("program")
            if not code:
                continue
            program = Program(task_id=task_id, prog_model=model, code=code)
            _insert_program(conn, program)
            _insert_generation(conn, task_id, model, trajectory.get("seed_index", 0),
                               program.program_id)
            written += 1
    conn.commit()
    return written


def import_file(conn: sqlite3.Connection, path: str, prog_model: str | None = None) -> int:
    """Import a JSONL file, dispatching on whether it holds PBTs or programs.

    Sniffs the first record: a `pbt_code` field marks a PBTs file, a `trajectories`
    field marks a programs file. Empty files import nothing.

    Args:
        conn: Open read-write connection to the store.
        path: Path to a pbt_data PBTs or trajectory JSONL file.
        prog_model: Model override forwarded to import_programs for files that do
            not name their model.

    Returns:
        The count returned by the chosen importer, or 0 for an empty file.

    Raises:
        ValueError: If the file's shape matches neither PBTs nor programs.
    """
    first = next(_read_jsonl(path), None)
    if first is None:
        return 0
    if "pbt_code" in first:
        return import_pbts(conn, path)
    if "trajectories" in first:
        return import_programs(conn, path, prog_model=prog_model)
    raise ValueError(f"cannot detect artifact kind for {path!r}: no pbt_code or trajectories")
