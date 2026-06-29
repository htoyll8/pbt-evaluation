"""APPS loader (codeparrot/apps).

APPS problems come in two forms, distinguished by the `input_output` JSON:
- stdio (the vast majority): the program reads stdin and writes stdout. Scored with
  io_mode="stdio" over [input, expected] pairs.
- call-based (a small minority, `fn_name` present): a LeetCode-style `class Solution` with
  a method. Scored with the function-assert path by instantiating Solution and asserting
  the method's return for each case.

Each task carries its `difficulty` (introductory / interview / competition) so the
canonical table can be sliced by difficulty.

Programs are executed at module scope (matching how a model writes a stdin script), so the
rare APPS reference solution that uses a top-level `return` (expecting the official harness's
function wrapping) is not supported; model outputs do not use that idiom.
"""
import json

from grading.datasets.base import Task, take

SOURCE = "codeparrot/apps"
_WHOLE_SPLIT = 10**9  # scan-all sentinel; take() caps it at the real split size


def _as_text(x: object) -> str:
    """Coerce an APPS stdin/stdout case to text.

    Most cases are strings; a few are list-wrapped (e.g. ['iloveyou']), which are
    flattened with newlines so they pipe through stdin cleanly.

    Args:
        x: A raw stdin/stdout case: a string, a list, or another value.

    Returns:
        The case as a single, newline-joined string.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(_as_text(e) for e in x)
    return str(x)


def _description(ex: dict) -> str:
    """Build the task description from an APPS example.

    Args:
        ex: A raw APPS record with a `question` and optional `starter_code`.

    Returns:
        The problem statement, with any starter code appended for the model to complete.
    """
    desc = ex["question"]
    starter = ex.get("starter_code") or ""
    if starter.strip():
        desc += "\n\nUse this starter code:\n" + starter
    return desc


def _call_tests(io_spec: dict) -> tuple[str, list[str]]:
    """Build the prelude and assert tests for a call-based problem.

    `candidate` is bound to the Solution method; each case asserts the method's return,
    accepting APPS's habit of wrapping the expected value in a single-element list.

    Args:
        io_spec: The parsed `input_output` spec, with `fn_name`, `inputs`, and `outputs`.

    Returns:
        A (prelude, tests) pair: code that binds `candidate`, and one assert per case.
    """
    prelude = f"_sol = Solution()\ncandidate = _sol.{io_spec['fn_name']}"
    # exp is the expected return, sometimes wrapped as [value]; accept either form.
    tests = [
        f"__exp = {exp!r}\n__got = candidate(*{inp!r})\nassert __got == __exp or [__got] == __exp"
        for inp, exp in zip(io_spec["inputs"], io_spec["outputs"])
    ]
    return prelude, tests


def load(n_tasks: int, difficulties: tuple[str, ...] | None = None) -> list[Task]:
    """Load up to n_tasks APPS tasks, scanning the split in order.

    The split is ordered interview-first, so reaching introductory or competition
    problems means scanning past the interview block. Problems with no usable tests
    are skipped.

    Args:
        n_tasks: Maximum number of tasks to collect.
        difficulties: Optional subset of {introductory, interview, competition} to keep.

    Returns:
        The collected APPS tasks (stdio or call-based), each tagged with its difficulty.
    """
    tasks: list[Task] = []
    skipped_no_tests = 0
    for ex in take(SOURCE, _WHOLE_SPLIT):  # whole split, lazily; we stop once we have n_tasks
        if len(tasks) >= n_tasks:
            break
        if difficulties and ex["difficulty"] not in difficulties:
            continue
        raw = ex["input_output"]
        io_spec = json.loads(raw) if raw else {}
        if not io_spec.get("inputs"):
            skipped_no_tests += 1  # no tests to score against; skip rather than log a fake pass
            continue
        task_id = str(ex["problem_id"])
        desc = _description(ex)
        if "fn_name" in io_spec:
            prelude, tests = _call_tests(io_spec)
            tasks.append(Task(task_id=task_id, description=desc, setup="", tests=tests,
                              prelude=prelude, difficulty=ex["difficulty"]))
        else:
            cases = [[_as_text(i), _as_text(o)] for i, o in zip(io_spec["inputs"], io_spec["outputs"])]
            tasks.append(Task(task_id=task_id, description=desc, setup="", tests=cases,
                              io_mode="stdio", difficulty=ex["difficulty"]))
    if skipped_no_tests:
        print(f"[apps] skipped {skipped_no_tests} problem(s) with no usable tests")
    return tasks
