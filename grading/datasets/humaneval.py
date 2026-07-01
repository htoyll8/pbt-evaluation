"""
HumanEval loader (openai/openai_humaneval).

HumanEval ships a `check(candidate)` body; we split it into individual asserts so each test scores separately (partial credit), and bind `candidate` to the entry point via the task prelude.
"""
from grading.datasets.base import Task, take

SOURCE = "openai/openai_humaneval"


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


def load(n_tasks: int) -> list[Task]:
    return [
        Task(
            task_id=str(ex["task_id"]),
            description=ex["prompt"],
            setup="",
            tests=extract_asserts(ex["test"]),
            prelude=f"candidate = {ex['entry_point']}",
            entry_point=ex["entry_point"],
            reference_solution=ex["prompt"] + ex["canonical_solution"],
        )
        for ex in take(SOURCE, n_tasks)
    ]
