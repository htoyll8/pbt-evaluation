"""
generate_seeds.py — generate candidate seed programs for PBT evaluation.

A code-generation model writes n candidate programs per task, each graded by the
vendored grading scorer (same harness as the self-correction experiment: function
name/arity hints for MBPP+, candidate binding for HumanEval, stdio for APPS,
partial-credit scoring in an isolated subprocess). Output is the single-attempt
trajectory format run_pbt_eval.py expects (refinement_attempts empty), so the
PBT experiment is decoupled from self-correction (fresh seeds) but graded
identically to it (shared harness).

Usage:
    python scripts/generate_seeds.py --dataset humaneval --model gpt-4o-mini --max_tasks 20
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Model
from tracking import track
from generate_pbts import extract_code
from grading.datasets import load_tasks
from grading.evaluators.scorer import make_scorer

os.makedirs("results", exist_ok=True)

# CLI dataset -> (grading benchmark name, difficulties filter)
DATASET_SPEC = {
    "mbppplus": ("mbppplus", None),
    "humaneval": ("humaneval", None),
    "apps_introductory": ("apps", ("introductory",)),
    "apps_competition": ("apps", ("competition",)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_SPEC), default="humaneval")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_tasks", type=int, default=50)
    parser.add_argument("--n", type=int, default=1, help="candidate seeds per task")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"results/seeds_{args.model}_{args.dataset}_n{args.n}.jsonl"

    name, difficulties = DATASET_SPEC[args.dataset]
    tasks = load_tasks(name, args.max_tasks, difficulties=difficulties)
    print(f"[INFO] {len(tasks)} {args.dataset} tasks via grading harness; generating with {args.model}")
    print(f"[INFO] Output → {args.out}")

    model = Model(model_name=args.model)

    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["task_id"] for line in open(args.out)}
        print(f"[INFO] Resuming — {len(done)} tasks already done")

    total_in = total_out = seeds = passes = 0
    with open(args.out, "a") as f:
        for i, task in enumerate(tasks):
            if task.task_id in done:
                continue
            programs, usage = model.generate(task.description, n=args.n)
            total_in += usage["input_tokens"]
            total_out += usage["output_tokens"]

            score = make_scorer(task.setup, task.tests, task.prelude,
                                task.per_timeout, task.io_mode)
            trajectories = []
            for s, raw in enumerate(programs):
                code = extract_code(raw)
                frac = score(code)            # fraction of tests passed, [0, 1]
                passed = frac == 1.0          # "passed" = all tests
                seeds += 1
                passes += 1 if passed else 0
                trajectories.append({
                    "seed_index": s,
                    "initial_program": code,
                    "initial_passed": passed,
                    "pass_fraction": frac,
                    "refinement_attempts": [],
                })

            f.write(json.dumps({
                "task_id": task.task_id,
                "dataset": args.dataset,
                "trajectories": trajectories,
                "k": args.n,
                "n_tests": task.n_tests,
                "generation_model": args.model,
                "usage": usage,
            }) + "\n")
            f.flush()

            n_pass = sum(1 for t in trajectories if t["initial_passed"])
            print(f"  [{i+1}/{len(tasks)}] {task.task_id} pass {n_pass}/{len(trajectories)} "
                  f"(tokens {total_in}in/{total_out}out)", flush=True)

    print(f"\nDone. {seeds} seeds, full-pass rate {passes}/{seeds} → {args.out}")
    track(
        experiment="pbt-seeds",
        run_name=f"{args.model}_{args.dataset}",
        params={"dataset": args.dataset, "model": args.model, "max_tasks": args.max_tasks, "n": args.n},
        metrics={
            "num_tasks": len(tasks),
            "num_seeds": seeds,
            "pass_rate": passes / max(seeds, 1),
            "input_tokens": total_in,
            "output_tokens": total_out,
        },
        artifact=args.out,
    )


if __name__ == "__main__":
    main()
