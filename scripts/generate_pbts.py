"""
generate_pbts.py

Generates Hypothesis property-based tests for MBPP+ or HumanEval tasks using an LLM.
Output JSONL is the persistent PBT library — reuse it across eval runs.

Usage:
    # MBPP+
    python scripts/generate_pbts.py \
        --dataset mbppplus \
        --model claude-sonnet-4-5 \
        --max_tasks 50 \
        --out pbt_data/pbts_claude_mbppplus_pilot.jsonl

    # HumanEval
    python scripts/generate_pbts.py \
        --dataset humaneval \
        --model claude-sonnet-4-5 \
        --max_tasks 20 \
        --out pbt_data/pbts_claude_humaneval_pilot.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Model
from tracking import track
from datasets import load_dataset

os.makedirs("pbt_data", exist_ok=True)


PBT_PROMPT_APPS = """\
You are writing a property-based test using the Python Hypothesis library for a competitive programming problem.

The solution is a complete stdin/stdout program. In the test, `fn` is a callable that accepts an input string
(everything that would be written to stdin) and returns the output string (everything the program writes to stdout).

## Problem
{prompt}

## Reference solution
```python
{code}
```

## Example I/O pairs (do NOT replicate as exact equality checks)
{unit_tests}

## Instructions
Write a Hypothesis `@given` test that:
1. Generates valid inputs as a formatted string (matching the input format described above)
2. Calls `fn(input_str)` and asserts PROPERTIES of the output string
3. Does NOT assert specific expected values — assert structural/mathematical invariants
4. Covers at least 2 distinct properties if possible

Good properties to consider:
- Output format (correct number of lines, each line is a valid integer, etc.)
- Output is within expected bounds (e.g. answer is non-negative, answer ≤ n)
- Idempotency (running the same input twice gives the same output)
- Monotonicity (larger input → larger/smaller output)
- Special cases (n=1 → specific output, empty → specific output)

Your response must be a single Python code block containing:
- All necessary imports
- A function named `test_pbt(fn)` where fn(input_str: str) -> str
- Inside `test_pbt`, define and immediately call a `@given(...)` inner function

Format:
```python
from hypothesis import given, settings, assume
from hypothesis import strategies as st

def test_pbt(fn):
    @given(...)
    @settings(max_examples=50)
    def _test(...):
        input_str = ...  # build the input string
        result = fn(input_str).strip()
        # assert properties of result here
    _test()
```

Only output the code block. Do not explain.
"""

PBT_PROMPT_NOREF = """\
You are writing a property-based test using the Python Hypothesis library for a function described below.

## Task
{prompt}

## Instructions
Write a Hypothesis `@given` test that:
1. Generates valid inputs using `hypothesis.strategies` (st)
2. Draws inputs that actually exercise the function: when arguments are related, draw them together (e.g. pick a search character from the string, not independently), using `st.data()` if needed
3. Asserts PROPERTIES of the output: invariants that must hold for ALL valid inputs
4. Calls the provided `fn` as the function under test; do not redefine that function at the top level

Good properties to consider:
1. The test oracle: implement a simple/brute-force version inside the test (a helper, not a top-level redefinition of the function) and assert `fn` agrees with it
2. There and back again: an inverse round-trips: decode(encode(x)) == x
3. Different paths, same destination: two routes give the same result
4. Some things never change: an invariant is preserved (permutation, sum, bounds)
5. Solve a smaller problem first: relate f(x) to f on a smaller input
6. Hard to prove, easy to verify: the result satisfies its defining condition
7. The more things change, the more they stay the same: idempotency: f(f(x)) == f(x)

Your response must be a single Python code block containing:
- All necessary imports (hypothesis, hypothesis.strategies as st, etc.)
- A function named `test_pbt(fn)` that takes the function under test as its only argument
- Inside `test_pbt`, define and immediately call a `@given(...)` inner function

Format:
```python
from hypothesis import given
from hypothesis import strategies as st

def test_pbt(fn):
    @given(...)
    def _test(...):
        result = fn(...)
        # assert properties here
    _test()
```

Only output the code block. Do not explain.
"""

PBT_PROMPT_NOREF_APPS = """\
You are writing a property-based test using the Python Hypothesis library for a competitive programming problem.

The solution is a complete stdin/stdout program. In the test, `fn` is a callable that accepts an input string
(everything that would be written to stdin) and returns the output string (everything the program writes to stdout).

## Problem
{prompt}

## Example I/O pairs (do NOT replicate as exact equality checks)
{unit_tests}

## Instructions
Write a Hypothesis `@given` test that:
1. Generates valid inputs as a formatted string (matching the input format described above)
2. Calls `fn(input_str)` and asserts PROPERTIES of the output string
3. Does NOT assert specific expected values — assert structural/mathematical invariants
4. Covers at least 2 distinct properties if possible

Good properties to consider:
- Output format (correct number of lines, each line is a valid integer, etc.)
- Output is within expected bounds (e.g. answer is non-negative, answer ≤ n)
- Idempotency (running the same input twice gives the same output)
- Monotonicity (larger input → larger/smaller output)
- Special cases (n=1 → specific output, empty → specific output)

Your response must be a single Python code block containing:
- All necessary imports
- A function named `test_pbt(fn)` where fn(input_str: str) -> str
- Inside `test_pbt`, define and immediately call a `@given(...)` inner function

Format:
```python
from hypothesis import given, settings, assume
from hypothesis import strategies as st

def test_pbt(fn):
    @given(...)
    @settings(max_examples=50)
    def _test(...):
        input_str = ...  # build the input string
        result = fn(input_str).strip()
        # assert properties of result here
    _test()
```

Only output the code block. Do not explain.
"""

PBT_PROMPT = """\
You are writing a property-based test using the Python Hypothesis library for a function described below.

## Task
{prompt}

## Reference solution
```python
{code}
```

## Unit tests (for reference only — do NOT replicate these as properties)
{unit_tests}

## Instructions
Write a Hypothesis `@given` test that:
1. Generates valid inputs using `hypothesis.strategies` (st)
2. Asserts PROPERTIES of the output — invariants that must hold for ALL valid inputs
3. Does NOT assert specific expected values (that would just re-implement a unit test)
4. Covers at least 2 distinct properties if possible

Good properties to consider:
- Output type and shape (result is always a list, length is non-negative, etc.)
- Boundary behaviour (empty input → specific output)
- Relationship between input and output (all elements of output appear in input)
- Monotonicity (larger input → larger/smaller output)
- Commutativity where applicable

Idempotency (`f(f(x)) == f(x)`) holds only for some functions, so assert it only when the
operation is genuinely idempotent AND the output is a valid input to the function. Many are not:
squaring is not idempotent, and a function returning a bool or int from a string cannot be
re-applied; asserting idempotency there would fail the correct solution.

Your response must be a single Python code block containing:
- All necessary imports (hypothesis, hypothesis.strategies as st, etc.)
- A function named `test_pbt(fn)` that takes the function under test as its only argument
- Inside `test_pbt`, define and immediately call a `@given(...)` inner function

Format:
```python
from hypothesis import given, settings, assume
from hypothesis import strategies as st

def test_pbt(fn):
    @given(...)
    @settings(max_examples=100)
    def _test(...):
        result = fn(...)
        # assert properties here
    _test()
```

Only output the code block. Do not explain.
"""

PBT_PROMPT_NOHINTS = """\
Write a Hypothesis property-based test for this function.

## Task
{prompt}

## Reference solution
```python
{code}
```

## Unit tests (for reference only — do NOT replicate these as properties)
{unit_tests}

Your response must be a single Python code block containing:
- A function named `test_pbt(fn)` that takes the function under test as its only argument
- Inside `test_pbt`, define and immediately call a `@given(...)` inner function

Only output the code block. Do not explain.
"""

PBT_PROMPT_NOREF_NOHINTS = """\
Write a Hypothesis property-based test for this function.

## Task
{prompt}

## Unit tests (for reference only — do NOT replicate these as properties)
{unit_tests}

Your response must be a single Python code block containing:
- A function named `test_pbt(fn)` that takes the function under test as its only argument
- Inside `test_pbt`, define and immediately call a `@given(...)` inner function

Only output the code block. Do not explain.
"""


PROPERTY_EXTRACT_PROMPT = """\
You are analyzing a Python function to identify correctness properties for property-based testing.

## Task
{prompt}

## Reference solution
```python
{code}
```

## Unit tests (for reference)
{unit_tests}

## Instructions
List the key correctness properties that any correct implementation of this function must satisfy.
For each property, write a short one-line description.
Output a bullet list of properties. Do not write code.
"""

PROPERTY_EXTRACT_PROMPT_NOREF = """\
You are analyzing a Python function to identify correctness properties for property-based testing.

## Task
{prompt}

## Unit tests (for reference)
{unit_tests}

## Instructions
List the key correctness properties that any correct implementation of this function must satisfy.
For each property, write a short one-line description.
Output a bullet list of properties. Do not write code.
"""

PBT_FROM_PROPERTIES_PROMPT = """\
You are writing a property-based test using the Python Hypothesis library for a function described below.

## Task
{prompt}

## Correctness properties to test
{properties}

## Instructions
Write a Hypothesis `@given` test that checks the properties listed above.
Generate valid inputs using `hypothesis.strategies` (st) and assert properties of the output.

Your response must be a single Python code block containing:
- All necessary imports (hypothesis, hypothesis.strategies as st, etc.)
- A function named `test_pbt(fn)` that takes the function under test as its only argument
- Inside `test_pbt`, define and immediately call a `@given(...)` inner function

Format:
```python
from hypothesis import given, settings, assume
from hypothesis import strategies as st

def test_pbt(fn):
    @given(...)
    @settings(max_examples=100)
    def _test(...):
        result = fn(...)
        # assert properties here
    _test()
```

Only output the code block. Do not explain.
"""


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?(.*?)```", text, flags=re.DOTALL)
    return blocks[0].strip() if blocks else text.strip()


def validate_pbt(code: str) -> bool:
    try:
        compile(code, "<pbt>", "exec")
    except SyntaxError:
        return False
    return "test_pbt" in code and "@given" in code


# ── Dataset adapters ──────────────────────────────────────────────────────────

def load_mbppplus(max_tasks: int) -> list[dict]:
    ds = load_dataset("evalplus/mbppplus")["test"]
    out = []
    for task in list(ds)[:max_tasks]:
        out.append({
            "task_id": task["task_id"],
            "prompt": task["prompt"].strip(),
            "code": task["code"].strip(),
            "unit_tests": "\n".join(task["test_list"][:3]),
        })
    return out


def load_apps(max_tasks: int, difficulty: str = "introductory") -> list[dict]:
    ds = load_dataset("codeparrot/apps", split="test[:5000]")
    ds = ds.filter(lambda ex: ex.get("difficulty") == difficulty)
    out = []
    for task in list(ds)[:max_tasks]:
        try:
            solutions = json.loads(task.get("solutions", "[]"))
            code = solutions[0].strip() if solutions else ""
        except Exception:
            code = ""
        try:
            io = json.loads(task.get("input_output", "{}"))
            pairs = list(zip(io.get("inputs", []), io.get("outputs", [])))[:3]
            unit_tests = "\n".join(
                f"Input:\n{inp.strip()}\nOutput:\n{out_.strip()}"
                for inp, out_ in pairs
            )
        except Exception:
            unit_tests = ""
        if not code:
            continue
        out.append({
            "task_id": task["problem_id"],
            "prompt": task["question"].strip(),
            "code": code,
            "unit_tests": unit_tests,
            "difficulty": difficulty,
            "pbt_type": "apps",  # signals runner to use stdin/stdout wrapper
        })
    return out


def load_humaneval(max_tasks: int) -> list[dict]:
    ds = load_dataset("openai/openai_humaneval")["test"]
    out = []
    for task in list(ds)[:max_tasks]:
        # Full solution = prompt (signature + docstring) + canonical_solution (body)
        full_code = task["prompt"].rstrip() + "\n" + task["canonical_solution"]
        # Extract assert lines from the check() function
        unit_tests = "\n".join(
            line.strip()
            for line in task["test"].splitlines()
            if line.strip().startswith("assert")
        )[:3 * 80]  # cap length
        out.append({
            "task_id": task["task_id"],
            "prompt": task["prompt"].strip(),
            "code": full_code.strip(),
            "unit_tests": unit_tests,
            "entry_point": task["entry_point"],
        })
    return out


# ── Generation ────────────────────────────────────────────────────────────────

def generate_pbt_two_step(model: Model, task: dict, retries: int = 2, no_ref: bool = False) -> dict:
    """Two-step: extract properties first, then generate PBT code from them."""
    # Step 1: Extract properties
    if no_ref:
        prop_prompt = PROPERTY_EXTRACT_PROMPT_NOREF.format(
            prompt=task["prompt"], unit_tests=task["unit_tests"])
    else:
        prop_prompt = PROPERTY_EXTRACT_PROMPT.format(
            prompt=task["prompt"], code=task["code"], unit_tests=task["unit_tests"])

    try:
        prop_text, usage1 = model.complete(prop_prompt, max_tokens=1024)
    except Exception as e:
        return {
            "task_id": task["task_id"], "dataset": task.get("dataset", "unknown"),
            "prompt": task["prompt"], "reference_code": task["code"],
            "entry_point": task.get("entry_point"),
            "pbt_code": None, "pbt_valid": False, "error": f"property extraction: {e}",
            "generation_model": model.model_name,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    # Step 2: Generate PBT from properties
    pbt_prompt = PBT_FROM_PROPERTIES_PROMPT.format(
        prompt=task["prompt"], properties=prop_text)

    for attempt in range(retries + 1):
        try:
            text, usage2 = model.complete(pbt_prompt, max_tokens=1024)
            pbt_code = extract_code(text)
            total_usage = {
                "input_tokens": usage1["input_tokens"] + usage2["input_tokens"],
                "output_tokens": usage1["output_tokens"] + usage2["output_tokens"],
            }
            return {
                "task_id": task["task_id"], "dataset": task.get("dataset", "unknown"),
                "prompt": task["prompt"], "reference_code": task["code"],
                "entry_point": task.get("entry_point"),
                "pbt_code": pbt_code, "pbt_valid": validate_pbt(pbt_code),
                "extracted_properties": prop_text,
                "generation_model": model.model_name, "usage": total_usage,
            }
        except Exception as e:
            if attempt == retries:
                return {
                    "task_id": task["task_id"], "dataset": task.get("dataset", "unknown"),
                    "prompt": task["prompt"], "reference_code": task["code"],
                    "entry_point": task.get("entry_point"),
                    "pbt_code": None, "pbt_valid": False, "error": str(e),
                    "generation_model": model.model_name,
                    "usage": {"input_tokens": usage1["input_tokens"], "output_tokens": usage1["output_tokens"]},
                }
            time.sleep(2)


def generate_pbt(model: Model, task: dict, retries: int = 2, no_ref: bool = False, no_hints: bool = False) -> dict:
    if task.get("pbt_type") == "apps":
        template = PBT_PROMPT_NOREF_APPS if no_ref else PBT_PROMPT_APPS
    elif no_hints:
        template = PBT_PROMPT_NOREF_NOHINTS if no_ref else PBT_PROMPT_NOHINTS
    else:
        template = PBT_PROMPT_NOREF if no_ref else PBT_PROMPT

    fmt_kwargs = {"prompt": task["prompt"], "unit_tests": task["unit_tests"]}
    if not no_ref:
        fmt_kwargs["code"] = task["code"]
    full_prompt = template.format(**fmt_kwargs)

    for attempt in range(retries + 1):
        try:
            text, usage = model.complete(full_prompt, max_tokens=1024)
            pbt_code = extract_code(text)
            return {
                "task_id": task["task_id"],
                "dataset": task.get("dataset", "unknown"),
                "prompt": task["prompt"],
                "reference_code": task["code"],
                "entry_point": task.get("entry_point"),
                "pbt_code": pbt_code,
                "pbt_valid": validate_pbt(pbt_code),
                "generation_model": model.model_name,
                "usage": usage,
            }
        except Exception as e:
            if attempt == retries:
                return {
                    "task_id": task["task_id"],
                    "dataset": task.get("dataset", "unknown"),
                    "prompt": task["prompt"],
                    "reference_code": task["code"],
                    "entry_point": task.get("entry_point"),
                    "pbt_code": None,
                    "pbt_valid": False,
                    "error": str(e),
                    "generation_model": model.model_name,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mbppplus", "humaneval", "apps_introductory", "apps_competition"], default="mbppplus")
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--max_tasks", type=int, default=50)
    parser.add_argument("--no-ref", action="store_true",
                        help="Generate PBTs WITHOUT reference solution (ablation condition)")
    parser.add_argument("--two-step", action="store_true",
                        help="Two-step: extract properties first, then generate PBT from them")
    parser.add_argument("--no-hints", action="store_true",
                        help="Remove 'Good properties to consider' hints from prompt")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        tags = []
        tags.append("noref" if args.no_ref else "withref")
        if args.two_step:
            tags.append("twostep")
        if args.no_hints:
            tags.append("nohints")
        args.out = f"pbt_data/pbts_{args.model}_{args.dataset}_{'_'.join(tags)}.jsonl"

    print(f"[INFO] Loading {args.dataset}...")
    if args.dataset == "mbppplus":
        tasks = load_mbppplus(args.max_tasks)
    elif args.dataset == "humaneval":
        tasks = load_humaneval(args.max_tasks)
    elif args.dataset == "apps_introductory":
        tasks = load_apps(args.max_tasks, difficulty="introductory")
    elif args.dataset == "apps_competition":
        tasks = load_apps(args.max_tasks, difficulty="competition")

    for t in tasks:
        t["dataset"] = args.dataset

    cond_parts = ["without-ref" if args.no_ref else "with-ref"]
    if args.two_step:
        cond_parts.append("two-step")
    if args.no_hints:
        cond_parts.append("no-hints")
    condition = ", ".join(cond_parts)
    print(f"[INFO] Generating PBTs for {len(tasks)} tasks using {args.model} ({condition})")
    print(f"[INFO] Output → {args.out}")

    model = Model(model_name=args.model)

    # Resume from existing output
    done = set()
    if Path(args.out).exists():
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add(r["task_id"])
        print(f"[INFO] Resuming — {len(done)} tasks already done")

    total_in = total_out = 0
    with open(args.out, "a") as f:
        for i, task in enumerate(tasks):
            if task["task_id"] in done:
                continue
            if args.two_step:
                result = generate_pbt_two_step(model, task, no_ref=args.no_ref)
            else:
                result = generate_pbt(model, task, no_ref=args.no_ref, no_hints=args.no_hints)
            result["condition"] = condition
            f.write(json.dumps(result) + "\n")
            f.flush()
            total_in += result["usage"]["input_tokens"]
            total_out += result["usage"]["output_tokens"]
            status = "✓" if result["pbt_valid"] else "✗"
            print(
                f"  [{i+1}/{len(tasks)}] {task['task_id']} {status}  "
                f"(tokens so far: {total_in}in / {total_out}out)",
                flush=True,
            )

    records = [json.loads(l) for l in open(args.out)]
    valid = sum(1 for r in records if r.get("pbt_valid"))
    print(f"\nDone. {valid}/{len(records)} valid PBTs → {args.out}")
    print(f"Total tokens: {total_in} input / {total_out} output")

    track(
        experiment="pbt-generation",
        run_name=f"{args.model}_{args.dataset}_{condition}",
        params={
            "dataset": args.dataset,
            "model": args.model,
            "max_tasks": args.max_tasks,
            "condition": condition,
            "no_ref": args.no_ref,
            "two_step": args.two_step,
            "no_hints": args.no_hints,
        },
        metrics={
            "num_tasks": len(records),
            "valid_pbts": valid,
            "valid_rate": valid / max(len(records), 1),
            "input_tokens": total_in,
            "output_tokens": total_out,
        },
        artifact=args.out,
    )


if __name__ == "__main__":
    main()
