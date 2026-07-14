"""
HumanEval loader (openai/openai_humaneval).

HumanEval ships a `check(candidate)` body; we split it into individual asserts so each test scores separately (partial credit), and bind `candidate` to the entry point via the task prelude.

Unlike MBPP+, HumanEval does not ship its augmented suite in the same dataset: the original
`check()` asserts are the weak suite, and the strengthened one lives in `evalplus/humanevalplus`.
We load that separately and align by task id, seeding it as the private suite. It is run whole
(all-or-nothing) with a longer timeout, since it is not a per-case loop we can split.
"""
from datasets import load_dataset

from grading.datasets.base import Task, take

SOURCE = "openai/openai_humaneval"
PLUS_SOURCE = "evalplus/humanevalplus"
PLUS_TIMEOUT = 30   # the whole plus harness runs as a single unit


def extract_asserts(test_str: str) -> list[str]:
    """Pull individual `assert` statements out of a check() body, joining bracket/paren continuations so a multi-line assert stays one statement."""
    lines = [ln.rstrip() for ln in test_str.splitlines()]
    asserts, i = [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("assert"):
            stmt = lines[i].strip()
            depth = stmt.count("(") - stmt.count(")") + stmt.count("[") - stmt.count("]")
            while depth > 0 and i + 1 < len(lines):
                i += 1
                stmt += " " + lines[i].strip()
                depth += lines[i].count("(") - lines[i].count(")") + lines[i].count("[") - lines[i].count("]")
            asserts.append(stmt)
        i += 1
    return asserts


def _load_plus() -> dict[str, str]:
    """task_id -> the whole HumanEval+ harness, with its check() call appended.

    Mirrors what scripts/mutation_eval.py does for its strong comparator, so the private
    suite and the mutation track's strong detector are the same tests.
    """
    return {
        str(ex["task_id"]): ex["test"] + f"\ncheck({ex['entry_point']})"
        for ex in load_dataset(PLUS_SOURCE)["test"]
    }


def load(n_tasks: int) -> list[Task]:
    plus = _load_plus()
    tasks = []
    for ex in take(SOURCE, n_tasks):
        tid = str(ex["task_id"])
        plus_harness = plus.get(tid)
        tasks.append(Task(
            task_id=tid,
            description=ex["prompt"],
            setup="",
            tests=extract_asserts(ex["test"]),
            prelude=f"candidate = {ex['entry_point']}",
            entry_point=ex["entry_point"],
            reference_solution=ex["prompt"] + ex["canonical_solution"],
            private_tests=[plus_harness] if plus_harness else [],
            private_timeout=PLUS_TIMEOUT,
        ))
    return tasks
