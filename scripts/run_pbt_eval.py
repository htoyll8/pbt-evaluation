"""
run_pbt_eval.py

Runs generated PBTs against solutions in an existing results JSONL file.
For MBPP+, also evaluates against the private test harness (`test` field).

Overfitting measures:
  overfit_pbt:     passes_unit=True AND passes_pbt=False
  overfit_private: passes_unit=True AND passes_private=False  (MBPP+ only)

Usage:
    python scripts/run_pbt_eval.py \
        --results results/results_claude-sonnet-4-5_mbppplus_interview_critique-refine_2026-04-09_01-41-05.jsonl \
        --pbts pbt_data/pbts_claude_mbppplus_pilot.jsonl \
        --out pbt_data/pbt_eval_claude_pilot.jsonl \
        --dataset mbppplus

    # HumanEval (no private tests):
    python scripts/run_pbt_eval.py \
        --results results/results_claude-sonnet-4-5_humaneval_*.jsonl \
        --pbts pbt_data/pbts_claude_humaneval_pilot.jsonl \
        --out pbt_data/pbt_eval_claude_humaneval_pilot.jsonl \
        --dataset humaneval
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracking import track

os.makedirs("pbt_data", exist_ok=True)


PBT_RUNNER_TEMPLATE = """\
import sys
from typing import *

{pbt_code}

# Solution under test
{solution_code}

# Run
try:
    test_pbt({fn_name})
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
"""

# Private test harness runner — injects solution then the test field verbatim.
# The test field directly calls the function by its hardcoded name (no check() wrapper).
PRIVATE_RUNNER_TEMPLATE = """\
import sys
from typing import *

{solution_code}

try:
{indented_test_code}
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
"""


def extract_fn_name(code: str) -> str:
    import re
    m = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else "solution"


def run_pbt_against_solution(pbt_code: str, solution_code: str, timeout: int = 30) -> dict:
    """Run a PBT against a solution in a subprocess. Returns {passed, output}."""
    fn_name = extract_fn_name(solution_code)
    runner = PBT_RUNNER_TEMPLATE.format(
        pbt_code=pbt_code,
        solution_code=solution_code,
        fn_name=fn_name,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(runner)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        passed = output.startswith("PASS")
        return {"passed": passed, "output": output[:500]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "TIMEOUT"}
    except Exception as e:
        return {"passed": False, "output": f"ERROR: {e}"}
    finally:
        os.unlink(tmp_path)


def run_private_test(test_code: str, solution_code: str, timeout: int = 30) -> dict:
    """Run the MBPP+ private test harness against a solution.

    The test field calls the function by its hardcoded name directly (no check() wrapper).
    We inject the solution then wrap the test body in try/except.
    """
    # Indent the test code so it sits inside the try block
    indented = "\n".join("    " + line for line in test_code.splitlines())
    runner = PRIVATE_RUNNER_TEMPLATE.format(
        solution_code=solution_code,
        indented_test_code=indented,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(runner)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        passed = output.startswith("PASS")
        return {"passed": passed, "output": output[:500]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "TIMEOUT"}
    except Exception as e:
        return {"passed": False, "output": f"ERROR: {e}"}
    finally:
        os.unlink(tmp_path)


def load_pbts(path: str) -> dict:
    pbts = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("pbt_valid") and r.get("pbt_code"):
                pbts[r["task_id"]] = r
    return pbts


def load_results(path: str) -> dict:
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results[r["task_id"]] = r
    return results


def load_private_tests(dataset: str) -> dict:
    """Load private test harnesses keyed by task_id. Returns {} if not applicable."""
    if dataset != "mbppplus":
        return {}
    try:
        from datasets import load_dataset
        ds = load_dataset("evalplus/mbppplus")["test"]
        private = {}
        for task in ds:
            private[task["task_id"]] = task["test"]  # the check(candidate) function string
        print(f"[INFO] Loaded private tests for {len(private)} MBPP+ tasks")
        return private
    except Exception as e:
        print(f"[WARN] Could not load private tests: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--pbts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dataset", choices=["mbppplus", "humaneval"], default="mbppplus")
    args = parser.parse_args()

    pbts = load_pbts(args.pbts)
    results = load_results(args.results)
    private_tests = load_private_tests(args.dataset)
    has_private = bool(private_tests)

    shared = set(pbts) & set(results)
    print(f"[INFO] PBTs: {len(pbts)}  |  Results: {len(results)}  |  Shared: {len(shared)}")
    if has_private:
        shared_private = shared & set(private_tests)
        print(f"[INFO] Tasks with private tests: {len(shared_private)}/{len(shared)}")

    # Resume
    done = set()
    if Path(args.out).exists():
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["task_id"], r["seed_index"], r["attempt"]))
        print(f"[INFO] Resuming — {len(done)} already evaluated")

    rows = []
    with open(args.out, "a") as f:
        for task_id in sorted(shared):
            pbt = pbts[task_id]
            task_result = results[task_id]
            check_code = private_tests.get(task_id) if has_private else None

            for traj in (task_result.get("trajectories") or []):
                if not traj or traj.get("initial_program") == "(timeout)":
                    continue
                seed_idx = traj.get("seed_index")

                # Evaluate initial program
                key = (task_id, seed_idx, 0)
                if key not in done:
                    init_prog = traj.get("initial_program", "")
                    unit_passed = traj.get("initial_passed", False)

                    pbt_result = run_pbt_against_solution(
                        pbt["pbt_code"], init_prog, timeout=args.timeout
                    )
                    row = {
                        "task_id": task_id,
                        "seed_index": seed_idx,
                        "attempt": 0,
                        "passes_unit": unit_passed,
                        "passes_pbt": pbt_result["passed"],
                        "overfit_pbt": unit_passed and not pbt_result["passed"],
                        "pbt_output": pbt_result["output"],
                    }
                    if has_private and check_code:
                        priv_result = run_private_test(check_code, init_prog, timeout=args.timeout)
                        row["passes_private"] = priv_result["passed"]
                        row["overfit_private"] = unit_passed and not priv_result["passed"]
                        row["private_output"] = priv_result["output"]

                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    rows.append(row)

                # Evaluate each refinement attempt
                for ra in (traj.get("refinement_attempts") or []):
                    attempt = ra.get("attempt", 1)
                    key = (task_id, seed_idx, attempt)
                    if key in done:
                        continue
                    prog = ra.get("program") or ra.get("refined_program") or ""
                    if not prog:
                        continue
                    unit_passed = ra.get("passed", False)

                    pbt_result = run_pbt_against_solution(
                        pbt["pbt_code"], prog, timeout=args.timeout
                    )
                    row = {
                        "task_id": task_id,
                        "seed_index": seed_idx,
                        "attempt": attempt,
                        "passes_unit": unit_passed,
                        "passes_pbt": pbt_result["passed"],
                        "overfit_pbt": unit_passed and not pbt_result["passed"],
                        "pbt_output": pbt_result["output"],
                    }
                    if has_private and check_code:
                        priv_result = run_private_test(check_code, prog, timeout=args.timeout)
                        row["passes_private"] = priv_result["passed"]
                        row["overfit_private"] = unit_passed and not priv_result["passed"]
                        row["private_output"] = priv_result["output"]

                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    rows.append(row)

    # Summary
    all_rows = [json.loads(l) for l in open(args.out)]
    unit_pass       = [r for r in all_rows if r["passes_unit"]]
    pbt_pass        = [r for r in all_rows if r["passes_pbt"]]
    overfit_pbt     = [r for r in all_rows if r.get("overfit_pbt")]

    print(f"\n{'='*55}")
    print(f"Results written to: {args.out}")
    print(f"\nTotal solution-attempts evaluated: {len(all_rows)}")
    print(f"  passes_unit:      {len(unit_pass):4d}  ({100*len(unit_pass)/max(len(all_rows),1):.1f}%)")
    print(f"  passes_pbt:       {len(pbt_pass):4d}  ({100*len(pbt_pass)/max(len(all_rows),1):.1f}%)")
    print(f"  overfit (PBT):    {len(overfit_pbt):4d}  ({100*len(overfit_pbt)/max(len(all_rows),1):.1f}%)")
    if unit_pass:
        print(f"  overfit_pbt rate among unit-passing: "
              f"{len(overfit_pbt)}/{len(unit_pass)} ({100*len(overfit_pbt)/len(unit_pass):.1f}%)")

    metrics = {
        "total_attempts": len(all_rows),
        "passes_unit": len(unit_pass),
        "passes_pbt": len(pbt_pass),
        "overfit_pbt": len(overfit_pbt),
    }
    if unit_pass:
        metrics["overfit_pbt_rate_unit"] = len(overfit_pbt) / len(unit_pass)
    private_rows = [r for r in all_rows if "passes_private" in r]
    if private_rows:
        metrics["passes_private"] = sum(1 for r in private_rows if r["passes_private"])
        metrics["overfit_private"] = sum(1 for r in all_rows if r.get("overfit_private"))

    track(
        experiment="pbt-eval",
        run_name=Path(args.results).stem,
        params={
            "results": args.results,
            "pbts": args.pbts,
            "dataset": args.dataset,
            "timeout": args.timeout,
        },
        metrics=metrics,
        artifact=args.out,
    )

    if has_private:
        priv_pass       = [r for r in all_rows if r.get("passes_private")]
        overfit_priv    = [r for r in all_rows if r.get("overfit_private")]
        overfit_both    = [r for r in all_rows if r.get("overfit_pbt") and r.get("overfit_private")]
        print(f"\n  --- Private test harness ---")
        print(f"  passes_private:   {len(priv_pass):4d}  ({100*len(priv_pass)/max(len(all_rows),1):.1f}%)")
        print(f"  overfit (priv):   {len(overfit_priv):4d}  ({100*len(overfit_priv)/max(len(all_rows),1):.1f}%)")
        if unit_pass:
            print(f"  overfit_priv rate among unit-passing: "
                  f"{len(overfit_priv)}/{len(unit_pass)} ({100*len(overfit_priv)/len(unit_pass):.1f}%)")
        print(f"  overfit (both):   {len(overfit_both):4d}")

    # Per-task breakdown
    task_stats = defaultdict(lambda: {"unit_pass": 0, "overfit_pbt": 0, "overfit_priv": 0})
    for r in all_rows:
        tid = r["task_id"]
        if r["passes_unit"]:
            task_stats[tid]["unit_pass"] += 1
        if r.get("overfit_pbt"):
            task_stats[tid]["overfit_pbt"] += 1
        if r.get("overfit_private"):
            task_stats[tid]["overfit_priv"] += 1

    overfit_pbt_tasks  = [t for t, v in task_stats.items() if v["overfit_pbt"] > 0]
    overfit_priv_tasks = [t for t, v in task_stats.items() if v["overfit_priv"] > 0]
    print(f"\nTasks with ≥1 overfit (PBT):     {len(overfit_pbt_tasks)}/{len(shared)}")
    if overfit_pbt_tasks:
        print("  Task IDs:", sorted(overfit_pbt_tasks)[:20])
    if has_private:
        print(f"Tasks with ≥1 overfit (private): {len(overfit_priv_tasks)}/{len(shared)}")
        if overfit_priv_tasks:
            print("  Task IDs:", sorted(overfit_priv_tasks)[:20])


if __name__ == "__main__":
    main()
