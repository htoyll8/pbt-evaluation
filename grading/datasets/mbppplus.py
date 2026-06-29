"""
MBPP+ loader (evalplus/mbppplus).

MBPP prompts don't name the function, so we infer it from the asserts and append a name/arity hint to the prompt.
"""
import re

from grading.datasets.base import Task, take

SOURCE = "evalplus/mbppplus"
PLUS_TIMEOUT = 30   # seconds for the whole-harness fallback (runs every case at once)

# The standard EvalPlus MBPP+ harness ends with this loop over (inputs, results).
_LOOP_RE = re.compile(r"for i, \(inp, exp\) in enumerate\(zip\(inputs, results\)\):")
_ASSERT_RE = re.compile(r"assertion\((\w+)\(\*inp\), exp, ([^\)]+)\)")

# builtins that can appear on the LHS of an assert but are not the task's function
WRAPPER_FUNCS = frozenset({"set", "sorted", "list", "tuple", "dict", "len", "max",
                           "min", "sum", "all", "any", "abs", "round", "int", "float", "str", "bool", "isclose"})


def _split_top_level_args(s: str) -> list[str]:
    args, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


def extract_function_signature(assert_lines: list[str]) -> tuple[str | None, int | None]:
    """Infer (function name, parameter count) from a task's assert statements."""
    for line in assert_lines:
        lhs = line.split("==")[0]
        i, func_start = 0, None
        while i < len(lhs):
            ch = lhs[i]
            if ch.isalpha() or ch == "_":
                if func_start is None:
                    func_start = i
            else:
                if func_start is not None:
                    name = lhs[func_start:i]
                    if i < len(lhs) and lhs[i] == "(":
                        paren, j = 1, i + 1
                        while j < len(lhs) and paren > 0:
                            if lhs[j] == "(":
                                paren += 1
                            elif lhs[j] == ")":
                                paren -= 1
                            j += 1
                        if paren == 0 and name not in WRAPPER_FUNCS:
                            return name, len(_split_top_level_args(lhs[i + 1:j - 1]))
                    func_start = None
            i += 1
    return None, None


def _build_desc(ex: dict, fname: str | None, pcount: int | None) -> str:
    """Task prompt augmented with the legacy MBPP function-name/arity hint.

    Args:
        ex: The raw MBPP+ record, supplying the base prompt.
        fname: Inferred function name, or None when none could be inferred.
        pcount: Inferred parameter count, or None when unknown.

    Returns:
        The prompt with the name and arity hints appended when available.
    """
    desc = ex["prompt"]
    if fname:
        desc += f"\nMAKE SURE THE FUNCTION IS NAMED `{fname}`."
    if pcount is not None:
        desc += f"\nTHE FUNCTION MUST TAKE EXACTLY {pcount} PARAMETER(S)."
    return desc


def _split_harness(test_src: str) -> tuple[str, list[str]] | None:
    """Split the EvalPlus `test` harness into (preamble, per-case test statements).

    The preamble defines `assertion`, `inputs`, `results`; each returned statement re-runs
    one case so the task is scored for partial credit. Returns None when the harness doesn't
    match the standard loop (caller falls back to whole-harness, all-or-nothing scoring).
    """
    m_loop = _LOOP_RE.search(test_src)
    m_assert = _ASSERT_RE.search(test_src)
    if not (m_loop and m_assert):
        return None
    preamble = test_src[:m_loop.start()]
    fname, atol = m_assert.group(1), m_assert.group(2)
    env: dict = {}
    try:  # run the preamble once to learn the case count
        exec(preamble, env)
        n = min(len(env["inputs"]), len(env["results"]))
    except Exception:
        return None
    tests = [f"assertion({fname}(*inputs[{i}]), results[{i}], {atol})" for i in range(n)]
    return preamble, tests


def load(n_tasks: int) -> list[Task]:
    """Load MBPP+ tasks scored against the full EvalPlus `test` harness.

    `test` is the augmented "plus" suite (the point of MBPP+), not the 3-assert base MBPP
    `test_list`. Where the harness matches the standard EvalPlus loop it is split into one
    test per case (partial credit); the few non-standard harnesses fall back to running the
    whole harness as a single all-or-nothing unit with a longer timeout.

    The harness preamble is run as `prelude` (after the candidate) so its `inputs`/`results`/
    `assertion` always win over any same-named variables the candidate happens to define.
    """
    tasks = []
    for ex in take(SOURCE, n_tasks):
        imports = ex["test_imports"]
        setup = "\n".join(imports) if isinstance(imports, list) else (imports or "")
        fname, pcount = extract_function_signature(ex["test_list"])  # name/arity from test_list
        desc = _build_desc(ex, fname, pcount)
        entry_point = fname or ""
        reference_solution = ex["code"]
        split = _split_harness(ex["test"])
        if split is not None:
            preamble, tests = split
            tasks.append(Task(task_id=str(ex["task_id"]), description=desc,
                              setup=setup, tests=tests, prelude=preamble,
                              entry_point=entry_point, reference_solution=reference_solution))
        else:
            tasks.append(Task(task_id=str(ex["task_id"]), description=desc,
                              setup=setup, tests=[ex["test"]], per_timeout=PLUS_TIMEOUT,
                              entry_point=entry_point, reference_solution=reference_solution))
    return tasks
