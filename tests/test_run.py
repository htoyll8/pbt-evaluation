"""End-to-end orchestrator test, fully offline.

Stubs the dataset loader and both model clients so run_pipeline exercises every
stage (seed, generate, validate, evaluate, analyze) with no network or API. Run:
    python -m pytest tests/test_run.py -q
    python tests/test_run.py
"""
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grading.datasets.base import Task as GradingTask
from pbt import db, run, seed


@contextmanager
def temp_db():
    """Yield a read-write connection to a throwaway database."""
    with tempfile.TemporaryDirectory() as d:
        conn = db.connect(os.path.join(d, "pbt.db"))
        try:
            yield conn
        finally:
            conn.close()


class FakeModel:
    """A stub model client returning canned text, recording prompts, no network."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt: str):
        """Return the canned reply and an empty usage dict."""
        self.prompts.append(prompt)
        return self.reply, {}


# An overfit program: it memorizes the one unit case and is wrong everywhere else.
_OVERFIT_PROGRAM = "```python\ndef add(a, b):\n    return 5 if (a, b) == (2, 3) else 0\n```"
# A correct, valid PBT: the reference solution passes it, the overfit program fails it.
_PBT = (
    "```python\n"
    "from hypothesis import given, strategies as st\n"
    "def test_pbt(candidate):\n"
    "    @given(st.integers(), st.integers())\n"
    "    def inner(a, b):\n"
    "        assert candidate(a, b) == a + b\n"
    "    inner()\n"
    "```"
)


def _stub_loader(monkeypatch):
    """Point seed.load_tasks at one in-memory task with a runnable reference solution."""
    task = GradingTask(
        task_id="2", description="add(a, b) returns a + b", setup="",
        tests=["assert candidate(2, 3) == 5"], prelude="candidate = add",
        entry_point="add", reference_solution="def add(a, b):\n    return a + b",
    )
    monkeypatch.setattr(seed, "load_tasks", lambda dataset, n_tasks: [task])


def test_run_pipeline_end_to_end_offline(monkeypatch):
    """The full pipeline catches a memorizing program with a valid PBT, offline."""
    _stub_loader(monkeypatch)
    with temp_db() as conn:
        metrics = run.run_pipeline(
            conn, "mbppplus", prog_model="prog-X", suite_model="suite-Y", n_tasks=1,
            prog_client=FakeModel(_OVERFIT_PROGRAM), suite_client=FakeModel(_PBT),
        )
    assert metrics["eligible"] == 1, "the overfit program passes the unit test, so it is eligible"
    assert metrics["caught"] == 1, "the valid PBT must catch the memorizing program"
    assert metrics["rate"] == 1.0, "one eligible pair, one catch"
    cell = metrics["matrix"][0]
    assert cell["prog_model"] == "prog-X" and cell["suite_model"] == "suite-Y", "matrix labels the cell"


def test_run_pipeline_is_idempotent(monkeypatch):
    """A second identical run rescues the cache and changes no verdict."""
    _stub_loader(monkeypatch)
    with temp_db() as conn:
        first = run.run_pipeline(conn, "mbppplus", "prog-X", "suite-Y", n_tasks=1,
                                 prog_client=FakeModel(_OVERFIT_PROGRAM), suite_client=FakeModel(_PBT))
        second = run.run_pipeline(conn, "mbppplus", "prog-X", "suite-Y", n_tasks=1,
                                  prog_client=FakeModel(_OVERFIT_PROGRAM), suite_client=FakeModel(_PBT))
        assert first["rate"] == second["rate"] == 1.0, "re-running yields the same verdict"


# Minimal monkeypatch shim so the file runs standalone, mirroring the other suites.
class _Patch:
    def setattr(self, obj, name, value):
        self._saved = getattr(obj, "_orig_load_tasks", None)
        obj._orig_load_tasks = getattr(obj, name)
        setattr(obj, name, value)


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        patch = _Patch()
        fn(patch)
        seed.load_tasks = seed._orig_load_tasks  # restore
        print(f"ok  {name}")
    print(f"\n{len(fns)} passed")
