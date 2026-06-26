"""Generation tests: programs and PBTs written into the store from a stub model.

No API calls and no scoring; a fake model returns canned text so generation is
deterministic and offline. Runs under pytest or standalone:

    python -m pytest tests/test_generate.py -q
    python tests/test_generate.py
"""
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbt import db, generate
from pbt.core import Task

CANNED_PROGRAM = "```python\ndef add(a, b):\n    return a + b\n```"
CANNED_PBT = (
    "```python\n"
    "from hypothesis import given, settings\n"
    "from hypothesis import strategies as st\n\n"
    "def test_pbt(fn):\n"
    "    @given(st.integers(), st.integers())\n"
    "    @settings(max_examples=10)\n"
    "    def _test(a, b):\n"
    "        assert fn(a, b) == fn(b, a)\n"
    "    _test()\n"
    "```"
)


class FakeModel:
    """A stub model that returns canned text and records every prompt it sees.

    Implements only the surface pbt.generate depends on: complete(prompt) returns
    a (text, usage) pair, so no network is ever touched.

    Attributes:
        text (str): The canned completion text returned for every prompt.
        model_name (str): Name mirrored from the real Model interface.
        prompts (list): Every prompt passed to complete, in call order.
    """

    def __init__(self, text: str, model_name: str = "fake"):
        self.text = text
        self.model_name = model_name
        self.prompts = []

    def complete(self, prompt, max_tokens=2048, temperature=None):
        """Record the prompt and return the canned (text, usage) pair."""
        self.prompts.append(prompt)
        return self.text, {"input_tokens": 0, "output_tokens": 0}


@contextmanager
def temp_db():
    """Yield a read-write connection to a throwaway database.

    Yields:
        An open connection to a temporary SQLite database with the schema created.
    """
    with tempfile.TemporaryDirectory() as d:
        conn = db.connect(os.path.join(d, "pbt.db"))
        try:
            yield conn
        finally:
            conn.close()


def _seed_task(conn: sqlite3.Connection, task: Task) -> None:
    """Insert a tasks row directly, ignoring an existing one with the same id."""
    conn.execute(
        "INSERT OR IGNORE INTO tasks(task_id, dataset, prompt, reference_solution, "
        "entry_point, setup, prelude, per_timeout, io_mode, difficulty) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (task.task_id, task.dataset, task.prompt, task.reference_solution,
         task.entry_point, task.setup, task.prelude, task.per_timeout, task.io_mode,
         task.difficulty),
    )


def _seed_unit_suite(conn: sqlite3.Connection, task_id: str) -> None:
    """Insert a benchmark unit suite so generate_pbts has hint material."""
    blob = json.dumps({"io_mode": "function", "tests": ["assert add(1, 2) == 3"]})
    conn.execute(
        "INSERT OR IGNORE INTO suites(suite_id, task_id, suite_model, kind, code) "
        "VALUES (?,?,?,?,?)",
        (f"unit-{task_id}", task_id, "benchmark", "unit", blob),
    )


def _add_task(conn: sqlite3.Connection) -> Task:
    """Seed one function-mode task plus its unit suite and return the task."""
    task = Task(task_id="t1", dataset="mbppplus", prompt="add two numbers",
                reference_solution="def add(a, b): return a + b", entry_point="add")
    _seed_task(conn, task)
    _seed_unit_suite(conn, task.task_id)
    conn.commit()
    return task


def test_generate_programs_writes_program_and_generation_rows():
    """A seeded task yields one program row and one generation row from the stub."""
    with temp_db() as conn:
        _add_task(conn)
        model = FakeModel(CANNED_PROGRAM)
        n = generate.generate_programs(conn, "mbppplus", "gpt-4o-mini", model=model)
        assert n == 1, "exactly one program should be written for one task, one sample"

        (progs,) = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        (gens,) = conn.execute("SELECT COUNT(*) FROM generations").fetchone()
        assert progs == 1, "one programs row expected"
        assert gens == 1, "one generations row expected"

        row = conn.execute("SELECT code, prog_model FROM programs").fetchone()
        assert "def add(a, b):" in row["code"], "extracted code must be the raw source"
        assert "```" not in row["code"], "backtick fences must be stripped by extract_code"
        assert row["prog_model"] == "gpt-4o-mini", "prog_model provenance must be recorded"


def test_generate_programs_records_multiplicity_per_sample():
    """Several samples of identical code give one program but several generation events."""
    with temp_db() as conn:
        _add_task(conn)
        model = FakeModel(CANNED_PROGRAM)
        n = generate.generate_programs(conn, "mbppplus", "gpt-4o-mini", n_samples=3,
                                       model=model)
        assert n == 1, "identical samples collapse to one new program row"

        (progs,) = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        (gens,) = conn.execute("SELECT COUNT(*) FROM generations").fetchone()
        assert progs == 1, "deduped to one program"
        assert gens == 3, "one generation event per sample_index"


def test_generate_pbts_writes_pbt_suite_with_returned_code():
    """A seeded task yields one kind='pbt' suite holding the raw PBT module."""
    with temp_db() as conn:
        _add_task(conn)
        model = FakeModel(CANNED_PBT)
        n = generate.generate_pbts(conn, "mbppplus", "claude-sonnet-4-5", model=model)
        assert n == 1, "exactly one PBT suite should be written for one task"

        row = conn.execute(
            "SELECT kind, code, suite_model FROM suites WHERE kind = 'pbt'"
        ).fetchone()
        assert row is not None, "a kind='pbt' suite must exist"
        assert row["suite_model"] == "claude-sonnet-4-5", "suite_model provenance recorded"
        assert "def test_pbt(fn):" in row["code"], "raw PBT module must follow the contract"
        assert "@given" in row["code"], "PBT must carry its @given property"
        assert "```" not in row["code"], "code is the raw module, not a fenced block"


def test_generate_pbts_includes_reference_for_ref_variant():
    """The 'ref' variant feeds the task's reference solution into the prompt."""
    with temp_db() as conn:
        _add_task(conn)
        model = FakeModel(CANNED_PBT)
        generate.generate_pbts(conn, "mbppplus", "claude-sonnet-4-5", model=model,
                               variant="ref")
        assert model.prompts, "the model must have been prompted"
        assert "def add(a, b): return a + b" in model.prompts[0], \
            "ref variant must embed the reference solution"


def test_regeneration_is_idempotent():
    """Re-running generation writes no new rows and returns zero."""
    with temp_db() as conn:
        _add_task(conn)
        prog_model = FakeModel(CANNED_PROGRAM)
        pbt_model = FakeModel(CANNED_PBT)

        first_progs = generate.generate_programs(conn, "mbppplus", "gpt-4o-mini",
                                                 model=prog_model)
        first_pbts = generate.generate_pbts(conn, "mbppplus", "claude-sonnet-4-5",
                                            model=pbt_model)
        assert first_progs == 1 and first_pbts == 1, "first run writes one of each"

        second_progs = generate.generate_programs(conn, "mbppplus", "gpt-4o-mini",
                                                  model=prog_model)
        second_pbts = generate.generate_pbts(conn, "mbppplus", "claude-sonnet-4-5",
                                             model=pbt_model)
        assert second_progs == 0, "re-running programs writes nothing new"
        assert second_pbts == 0, "re-running PBTs writes nothing new"

        (progs,) = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        (pbts,) = conn.execute("SELECT COUNT(*) FROM suites WHERE kind='pbt'").fetchone()
        (gens,) = conn.execute("SELECT COUNT(*) FROM generations").fetchone()
        assert progs == 1, "still one program after re-run"
        assert pbts == 1, "still one PBT suite after re-run"
        assert gens == 1, "still one generation event after re-run"


def test_unknown_variant_raises():
    """An unknown PBT variant is rejected before any model call."""
    with temp_db() as conn:
        _add_task(conn)
        model = FakeModel(CANNED_PBT)
        raised = False
        try:
            generate.generate_pbts(conn, "mbppplus", "claude", model=model,
                                   variant="bogus")
        except ValueError:
            raised = True
        assert raised, "an unknown variant must raise ValueError"
        assert not model.prompts, "no model call should happen for a bad variant"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
