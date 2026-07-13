"""Mutation-based overfit-catch eval.

For each task we mutate the canonical solution and keep the mutants that slip past the WEAK
public tests (the gate): synthetic overfit programs that look correct under the public suite.
Over that gated set we then measure two comparators, the STRONG private tests (plus) and each
valid PBT. The headline number is the PBT catch rate (pbt_caught / base_passing); plus is a
reference baseline, not a gate condition.

Weak gate and strong comparator by dataset:
  mbppplus  : weak = base `test_list`; strong = the seeded plus harness (kind='unit' suites).
  humaneval : weak = original HumanEval `check()` asserts; strong = the HumanEval+ harness
              (evalplus/humanevalplus), because the seeded unit suite here is itself the weak
              set, so the strong comparator can't come from the store.

Usage: python scripts/mutation_eval.py <db> [--dataset mbppplus|humaneval] [--limit 20]
"""
import argparse
import json
import sqlite3

from datasets import load_dataset

from grading.datasets.humaneval import extract_asserts
from grading.evaluators.scorer import make_scorer
from pbt.core.score import run_property
from pbt.mutation import generate_mutants


def load_base_tests(dataset: str) -> dict:
    """task_id -> (base tests, setup, prelude): the WEAK public gate a mutant must slip past.

    The gate is the benchmark's original (pre-augmentation) tests:
      mbppplus  : the 3-assert MBPP base `test_list`; the function is referenced by name, so no
                  prelude is needed (setup carries `test_imports`).
      humaneval : the original HumanEval `check()` asserts, which reference `candidate`, so the
                  prelude binds it to the task's entry point (mirroring grading/datasets/humaneval.py).
    """
    if dataset == "mbppplus":
        ds = load_dataset("evalplus/mbppplus")["test"]
        out = {}
        for ex in ds:
            imports = ex["test_imports"]
            setup = "\n".join(imports) if isinstance(imports, list) else (imports or "")
            out[str(ex["task_id"])] = (list(ex["test_list"]), setup, "")
        return out
    if dataset == "humaneval":
        ds = load_dataset("openai/openai_humaneval")["test"]
        return {
            str(ex["task_id"]): (extract_asserts(ex["test"]), "", f"candidate = {ex['entry_point']}")
            for ex in ds
        }
    raise ValueError(f"base gate is wired for mbppplus and humaneval only, not {dataset!r}")


def load_plus_tests(dataset: str) -> dict | None:
    """task_id -> [strong-harness test string], the STRONG comparator, or None to fall back to
    the seeded kind='unit' suites.

    MBPP+'s strong tests are already in the db (its loader seeds the plus harness as the unit
    suite), so returning None reuses them. HumanEval's are not: its unit suite is the weak gate
    itself, so reading it back would give a comparator identical to the gate (plus_caught always
    0). Instead we fetch the strong set fresh from HumanEval+ (evalplus/humanevalplus `test`) and
    run it whole, appending the `check(entry_point)` call, all-or-nothing.
    """
    if dataset != "humaneval":
        return None
    ds = load_dataset("evalplus/humanevalplus")["test"]
    return {str(ex["task_id"]): [ex["test"] + f"\ncheck({ex['entry_point']})"] for ex in ds}


def _passes(setup: str, tests: list, prelude: str, program: str, per_timeout: int = 5) -> bool:
    return make_scorer(setup, tests, prelude=prelude, io_mode="function",
                       per_timeout=per_timeout)(program) == 1.0


def main(db: str, dataset: str, limit: int) -> None:
    base = load_base_tests(dataset)
    c = sqlite3.connect(db)
    c.execute(
        "CREATE TABLE IF NOT EXISTS mutation_results ("
        "task_id TEXT, suite_model TEXT, n_base_passing INTEGER, "
        "plus_caught INTEGER, pbt_caught INTEGER, "
        "PRIMARY KEY (task_id, suite_model))"
    )
    # Strong comparator: reuse the seeded plus harness (mbppplus) unless the dataset loads its
    # strong set from elsewhere (humaneval -> HumanEval+); see load_plus_tests.
    strong = load_plus_tests(dataset)
    plus_by_task = strong if strong is not None else {
        tid: json.loads(code)["tests"]
        for tid, code in c.execute("select task_id, code from suites where kind='unit'")
    }
    plus_timeout = 30 if dataset == "humaneval" else 5  # HumanEval+ runs the whole harness at once
    tasks = c.execute("select task_id, reference_solution, entry_point, setup, prelude from tasks").fetchall()

    total_overfit_pairs = total_killed = total_plus = 0
    rows = []
    for tid, ref, ep, setup, prelude in tasks:
        if tid not in base:
            continue
        base_tests, base_setup, base_prelude = base[tid]
        plus_tests = plus_by_task.get(tid, [])
        if not plus_tests:
            continue
        # gate: mutants that pass the WEAK base tests (slip past weak public).
        # plus and the PBT are measured comparators over this same set, not gate conditions.
        base_passing = []
        plus_caught = 0
        for m in generate_mutants(ref, limit=limit):
            try:
                if not _passes(base_setup, base_tests, base_prelude, m):
                    continue  # caught by weak public; not an overfit candidate
                base_passing.append(m)
                if not _passes(setup or "", plus_tests, prelude or "", m, per_timeout=plus_timeout):
                    plus_caught += 1  # plus (strong private) catches it
            except Exception:
                continue
        if not base_passing:
            continue
        pbts = c.execute("select suite_model, code from suites where kind='pbt' and task_id=? and valid=1", (tid,)).fetchall()
        for sm, pbt in pbts:
            pbt_caught = sum(1 for m in base_passing if not run_property(m, pbt, ep))
            rows.append((tid, sm, len(base_passing), plus_caught, pbt_caught))
            total_overfit_pairs += len(base_passing)
            total_killed += pbt_caught
            total_plus += plus_caught

    c.executemany(
        "INSERT OR REPLACE INTO mutation_results "
        "(task_id, suite_model, n_base_passing, plus_caught, pbt_caught) VALUES (?,?,?,?,?)",
        rows,
    )
    c.commit()

    print(f"{'task':6}{'suite_model':14}{'base_pass':>10}{'plus_catch':>11}{'pbt_catch':>10}")
    for tid, sm, n, pc, kc in rows:
        print(f"{tid:6}{sm:14}{n:>10}{pc:>11}{kc:>10}")
    print("-"*51)
    if total_overfit_pairs:
        print(f"over {total_overfit_pairs} base-passing mutant-PBT pairs:")
        print(f"  plus (private) catch: {total_plus}/{total_overfit_pairs} = {total_plus/total_overfit_pairs:.3f}")
        print(f"  PBT          catch:   {total_killed}/{total_overfit_pairs} = {total_killed/total_overfit_pairs:.3f}")
    else:
        print("no base-passing mutants found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--dataset", default="mbppplus")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    main(args.db, args.dataset, args.limit)
