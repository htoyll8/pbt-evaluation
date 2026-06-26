"""Importer tests: PBTs become pbt suites, trajectories become programs, re-import no-ops.

No API calls and no real data files; each test writes a tiny JSONL fixture to a temp
path. Runs under pytest or standalone:

    python -m pytest tests/test_importer.py -q
    python tests/test_importer.py
"""
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbt import db
from pbt.core import Task
from pbt import importer


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


def _seed_task(conn: sqlite3.Connection, task_id: str) -> None:
    """Insert a minimal task row so imported artifacts satisfy the foreign key."""
    t = Task(task_id=task_id, dataset="mbppplus", prompt="p")
    conn.execute(
        "INSERT OR IGNORE INTO tasks(task_id, dataset, prompt, reference_solution, setup, "
        "prelude, per_timeout, io_mode, difficulty) VALUES (?,?,?,?,?,?,?,?,?)",
        (t.task_id, t.dataset, t.prompt, t.reference_solution, t.setup, t.prelude,
         t.per_timeout, t.io_mode, t.difficulty),
    )
    conn.commit()


def _write_jsonl(records: list) -> str:
    """Write records as a JSONL file and return its path (caller cleans up)."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


_PBT_CODE = (
    "from hypothesis import given\n"
    "from hypothesis import strategies as st\n"
    "def test_pbt(fn):\n"
    "    @given(st.integers())\n"
    "    def inner(x):\n"
    "        assert fn(x) == fn(x)\n"
    "    inner()\n"
)


def test_import_pbts_creates_pbt_suites():
    """Importing a PBTs file writes kind='pbt' suites with raw code and model."""
    with temp_db() as conn:
        _seed_task(conn, "2")
        path = _write_jsonl([
            {"task_id": 2, "dataset": "mbppplus", "pbt_code": _PBT_CODE,
             "pbt_valid": True, "generation_model": "claude-sonnet-4-5"},
        ])
        try:
            n = importer.import_pbts(conn, path)
        finally:
            os.remove(path)
        assert n == 1, "one PBT record should produce one suite"
        row = conn.execute("SELECT task_id, suite_model, kind, code, valid FROM suites").fetchone()
        assert row["kind"] == "pbt", "imported PBT must be stored as kind='pbt'"
        assert row["task_id"] == "2", "integer task_id must be coerced to the seeded string id"
        assert row["suite_model"] == "claude-sonnet-4-5", "suite_model must be the authoring model"
        assert row["code"] == _PBT_CODE, "PBT code must be stored raw, not JSON-wrapped"
        assert row["valid"] == 1, "pbt_valid=True must map onto the suite validity flag"


def test_import_pbts_skips_unseeded_task():
    """A PBT whose task was never seeded is skipped, leaving suites empty."""
    with temp_db() as conn:
        path = _write_jsonl([
            {"task_id": 999, "pbt_code": _PBT_CODE, "generation_model": "m"},
        ])
        try:
            n = importer.import_pbts(conn, path)
        finally:
            os.remove(path)
        assert n == 0, "import must skip records referencing an unseeded task"
        (count,) = conn.execute("SELECT COUNT(*) FROM suites").fetchone()
        assert count == 0, "no suite row may violate the foreign key"


def test_import_programs_creates_program_and_generation_rows():
    """Importing trajectories writes deduped programs and a generation per seed."""
    with temp_db() as conn:
        _seed_task(conn, "2")
        path = _write_jsonl([
            {"task_id": 2, "dataset": "mbppplus", "generation_model": "gpt-5.1",
             "trajectories": [
                 {"seed_index": 1, "initial_program": "def f(x):\n    return x"},
                 {"seed_index": 2, "initial_program": "def f(x):\n    return x"},
                 {"seed_index": 3, "initial_program": "def g(x):\n    return x + 1"},
             ]},
        ])
        try:
            n = importer.import_programs(conn, path)
        finally:
            os.remove(path)
        assert n == 3, "every trajectory should be counted"
        (progs,) = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        assert progs == 2, "identical code across seeds must dedup to one program"
        (gens,) = conn.execute("SELECT COUNT(*) FROM generations").fetchone()
        assert gens == 3, "each seed draw must record its own generation event"
        model = conn.execute("SELECT DISTINCT prog_model FROM programs").fetchone()[0]
        assert model == "gpt-5.1", "prog_model must come from the file's generation_model"


def test_import_programs_uses_model_override():
    """A file lacking a model uses the prog_model override for every program."""
    with temp_db() as conn:
        _seed_task(conn, "2")
        path = _write_jsonl([
            {"task_id": 2, "dataset": "mbppplus", "trajectories": [
                {"seed_index": 1, "initial_program": "def f(x):\n    return x"},
            ]},
        ])
        try:
            n = importer.import_programs(conn, path, prog_model="claude")
        finally:
            os.remove(path)
        assert n == 1, "the single trajectory should import under the override model"
        model = conn.execute("SELECT prog_model FROM programs").fetchone()[0]
        assert model == "claude", "prog_model override must label the imported program"


def test_reimport_is_idempotent():
    """Re-importing the same PBTs and programs adds no new rows."""
    with temp_db() as conn:
        _seed_task(conn, "2")
        pbt_path = _write_jsonl([
            {"task_id": 2, "pbt_code": _PBT_CODE, "pbt_valid": True,
             "generation_model": "claude"},
        ])
        prog_path = _write_jsonl([
            {"task_id": 2, "generation_model": "gpt-5.1", "trajectories": [
                {"seed_index": 1, "initial_program": "def f(x):\n    return x"},
            ]},
        ])
        try:
            importer.import_pbts(conn, pbt_path)
            importer.import_pbts(conn, pbt_path)  # re-import must no-op
            importer.import_programs(conn, prog_path)
            importer.import_programs(conn, prog_path)  # re-import must no-op
        finally:
            os.remove(pbt_path)
            os.remove(prog_path)
        (suites,) = conn.execute("SELECT COUNT(*) FROM suites").fetchone()
        (progs,) = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        (gens,) = conn.execute("SELECT COUNT(*) FROM generations").fetchone()
        assert suites == 1, "content-hash PK must dedup the re-imported suite"
        assert progs == 1, "content-hash PK must dedup the re-imported program"
        assert gens == 1, "deterministic gen_id must dedup the re-imported generation"


def test_import_file_dispatches_on_shape():
    """import_file routes a pbt_code file to PBTs and a trajectories file to programs."""
    with temp_db() as conn:
        _seed_task(conn, "2")
        pbt_path = _write_jsonl([
            {"task_id": 2, "pbt_code": _PBT_CODE, "generation_model": "claude"},
        ])
        prog_path = _write_jsonl([
            {"task_id": 2, "generation_model": "gpt-5.1", "trajectories": [
                {"seed_index": 1, "initial_program": "def f(x):\n    return x"},
            ]},
        ])
        try:
            n_pbt = importer.import_file(conn, pbt_path)
            n_prog = importer.import_file(conn, prog_path)
        finally:
            os.remove(pbt_path)
            os.remove(prog_path)
        assert n_pbt == 1, "a file with pbt_code must dispatch to import_pbts"
        assert n_prog == 1, "a file with trajectories must dispatch to import_programs"
        (suites,) = conn.execute("SELECT COUNT(*) FROM suites WHERE kind='pbt'").fetchone()
        (progs,) = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        assert suites == 1 and progs == 1, "each file must land in its own table"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
