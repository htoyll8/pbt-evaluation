"""Isolated test worker (benchmark-agnostic). Reads {setup, program, prelude, tests,
per_timeout, io_mode} as JSON on stdin and prints "##LCT##" + {"passed": int, "total": int}.
Runs in its own process group so the parent can SIGKILL the whole tree as a backstop.

Two scoring modes:
- "function" (default): exec setup+program(+prelude, e.g. binding `candidate`), then run
  each test as an assert with a per-assert SIGALRM timeout (partial credit survives one
  hang). `tests` is a list of assert-statement strings.
- "stdio": for each [input, expected] case, run the program fresh with `input` piped to
  stdin, capture stdout, and compare normalized (trailing whitespace stripped, as APPS
  does). `tests` is a list of [input, expected] pairs.
"""
import io
import json
import signal
import sys
import time


class _Timeout(Exception):
    pass


def _handler(signum, frame) -> None:
    raise _Timeout()


def _normalize(s: str) -> str:
    """APPS-style stdout comparison: ignore trailing whitespace per line and trailing blank lines."""
    return "\n".join(line.rstrip() for line in s.rstrip().splitlines())


def _over_budget(start: float, budget: float | None) -> bool:
    """True once the worker's own deadline passes, so it can stop and still emit a result
    (partial credit) before the parent SIGKILLs it at the slightly-larger overall timeout."""
    return budget is not None and (time.monotonic() - start) > budget


def _run_function(setup: str, program: str, prelude: str, tests: list, per: int,
                  budget: float | None) -> list[bool]:
    """Per-case pass list (length len(tests)); cases left unrun by the budget are False."""
    env: dict = {}
    try:
        exec(setup or "", env)
        exec(program, env)
        if prelude:
            exec(prelude, env)
    except BaseException:  # incl. SystemExit from candidate code
        return [False] * len(tests)
    start, cases = time.monotonic(), []
    for t in tests:
        if _over_budget(start, budget):
            break  # remaining cases stay False (padded below)
        try:
            signal.alarm(per)
            exec(t, env)
            cases.append(True)
        except BaseException:  # _Timeout, AssertionError, SystemExit, etc.
            cases.append(False)
        finally:
            signal.alarm(0)
    return cases + [False] * (len(tests) - len(cases))


def _run_stdio(setup: str, program: str, tests: list, per: int, budget: float | None) -> list[bool]:
    """Run the program once per case with piped stdin; per-case True when stdout matches.
    Cases left unrun by the budget stay False."""
    start, cases = time.monotonic(), []
    for case in tests:
        if _over_budget(start, budget):
            break  # remaining cases stay False (padded below)
        stdin_text, expected = case[0], case[1]
        saved_in, saved_out = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = io.StringIO(stdin_text), io.StringIO()
        # __name__ == "__main__" so a solution guarded by `if __name__ == "__main__":`
        # actually runs its driver (common for stdin scripts); without this it scores 0.
        env: dict = {"__name__": "__main__"}
        try:
            signal.alarm(per)
            exec(setup or "", env)
            exec(program, env)  # reads input()/sys.stdin, writes via print()
            got = sys.stdout.getvalue()
        except BaseException:  # _Timeout, SystemExit, runtime errors
            got = None
        finally:
            signal.alarm(0)
            sys.stdin, sys.stdout = saved_in, saved_out
        cases.append(got is not None and _normalize(got) == _normalize(expected))
    return cases + [False] * (len(tests) - len(cases))


def main() -> None:
    data = json.load(sys.stdin)
    setup, program, tests = data["setup"], data["program"], data["tests"]
    prelude = data.get("prelude", "")
    per = int(data.get("per_timeout", 5))
    io_mode = data.get("io_mode", "function")
    budget = data.get("budget")  # worker's own deadline (seconds); None => no self-limit

    signal.signal(signal.SIGALRM, _handler)
    if io_mode == "stdio":
        cases = _run_stdio(setup, program, tests, per, budget)
    elif io_mode == "function":
        cases = _run_function(setup, program, prelude, tests, per, budget)
    else:
        raise ValueError(f"unknown io_mode {io_mode!r}")
    # `cases` is the per-case pass vector; passed/total stay for existing callers.
    print("##LCT##" + json.dumps({"passed": sum(cases), "total": len(tests), "cases": cases}))


if __name__ == "__main__":
    main()
