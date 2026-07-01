"""The single source of truth: an append-only, immutable SQLite store.

Five normalized tables keyed by content-hash IDs. Writers use INSERT OR IGNORE so re-running is a no-op (idempotent), never an overwrite. Analysis opens the file read-only (mode=ro) so it physically cannot mutate the facts.

    tasks        one benchmark problem + its grading material
    programs     deduped candidate code, keyed by content hash
    generations  one row per sampling event (GROUP BY program_id for multiplicity)
    suites       deduped tests of any kind (unit | pbt | private)
    results      outcome of score(program, suite): {passed, pass_fraction}

results' primary key (program_id, suite_id) is the score() cache: a row's existence
means "already scored," so evaluate skips it on re-run.

Attributes:
    DEFAULT_PATH (str): Default path to the SQLite database file.
    SCHEMA (str): DDL run on connect; creates the tables and indexes if absent.
"""
import sqlite3

DEFAULT_PATH = "pbt.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id            TEXT PRIMARY KEY,
    dataset            TEXT NOT NULL,
    prompt             TEXT NOT NULL,
    reference_solution TEXT,                       -- for the PBT validity filter
    entry_point        TEXT NOT NULL DEFAULT '',   -- name of the task's candidate function
    setup              TEXT NOT NULL DEFAULT '',
    prelude            TEXT NOT NULL DEFAULT '',
    per_timeout        INTEGER NOT NULL DEFAULT 5,
    io_mode            TEXT NOT NULL DEFAULT 'function',
    difficulty         TEXT NOT NULL DEFAULT 'na'
);

CREATE TABLE IF NOT EXISTS programs (
    program_id TEXT PRIMARY KEY,                    -- hash(task_id, prog_model, code)
    task_id    TEXT NOT NULL REFERENCES tasks(task_id),
    prog_model TEXT NOT NULL,
    code       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    gen_id       TEXT PRIMARY KEY,                  -- hash(task_id, prog_model, sample_index)
    task_id      TEXT NOT NULL REFERENCES tasks(task_id),
    prog_model   TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    program_id   TEXT NOT NULL REFERENCES programs(program_id)
);

CREATE TABLE IF NOT EXISTS suites (
    suite_id    TEXT PRIMARY KEY,                   -- hash(task_id, suite_model, code)
    task_id     TEXT NOT NULL REFERENCES tasks(task_id),
    suite_model TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('unit', 'pbt', 'private')),
    code        TEXT NOT NULL,
    valid       INTEGER                             -- NULL=unchecked, 0/1 after validity filter
);

CREATE TABLE IF NOT EXISTS results (
    program_id    TEXT NOT NULL REFERENCES programs(program_id),
    suite_id      TEXT NOT NULL REFERENCES suites(suite_id),
    task_id       TEXT NOT NULL,
    prog_model    TEXT NOT NULL,                    -- denormalized for one-table analysis queries
    suite_model   TEXT NOT NULL,
    kind          TEXT NOT NULL,
    passed        INTEGER NOT NULL,                 -- 0/1
    pass_fraction REAL NOT NULL,
    PRIMARY KEY (program_id, suite_id)              -- the score() cache key
);

CREATE INDEX IF NOT EXISTS idx_programs_task   ON programs(task_id, prog_model);
CREATE INDEX IF NOT EXISTS idx_suites_task     ON suites(task_id, suite_model, kind);
CREATE INDEX IF NOT EXISTS idx_generations_pid ON generations(program_id);
CREATE INDEX IF NOT EXISTS idx_results_task    ON results(task_id, kind);
"""


def connect(path: str = DEFAULT_PATH) -> sqlite3.Connection:
    """Open a read-write connection and ensure the schema exists.

    Enables foreign keys and WAL journaling so that multiple tasks can append
    concurrently within a run without corrupting the file.

    Args:
        path: Filesystem path to the SQLite database; created if it does not exist.

    Returns:
        An open read-write connection whose rows are accessible by column name.
    """
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def connect_ro(path: str = DEFAULT_PATH) -> sqlite3.Connection:
    """Open a read-only connection for analysis.

    Any write raises sqlite3.OperationalError, so analysis code physically cannot
    mutate the source of truth.

    Args:
        path: Filesystem path to an existing SQLite database.

    Returns:
        An open read-only connection whose rows are accessible by column name.

    Raises:
        sqlite3.OperationalError: If the database file does not exist.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn
