"""Mutation-based overfit-catch eval.

For each task we mutate the canonical solution and keep the mutants that pass the WEAK
public tests (MBPP+ base test_list) but fail the STRONG private tests (the plus harness):
those are synthetic overfit programs. We then ask, for each valid PBT, what fraction of
that overfit set it catches (a synthetic overfit-catch rate). The plus harness catches all
of them by construction; the question is how close a single generated property gets.

Usage: python scripts/mutation_eval.py <db> [--dataset mbppplus] [--limit 20]
"""
import argparse
import json
import sqlite3

from datasets import load_dataset

from grading.evaluators.scorer import make_scorer
from pbt.core.score import run_property
from pbt.mutation import generate_mutants


def load_base_tests(dataset: str) -> dict:
    """task_id -> (base test_list asserts, setup imports) from the raw EvalPlus dataset."""
    if dataset != "mbppplus":
        raise ValueError("base test_list gate is wired for mbppplus only")
    ds = load_dataset("evalplus/mbppplus")["test"]
    out = {}
    for ex in ds:
        imports = ex["test_imports"]
        setup = "\n".join(imports) if isinstance(imports, list) else (imports or "")
        out[str(ex["task_id"])] = (list(ex["test_list"]), setup)
    return out


def _passes(setup: str, tests: list, prelude: str, program: str) -> bool:
    return make_scorer(setup, tests, prelude=prelude, io_mode="function")(program) == 1.0


def main(db: str, dataset: str, limit: int) -> None:
    base = load_base_tests(dataset)
    c = sqlite3.connect(db)
    c.execute(
        "CREATE TABLE IF NOT EXISTS mutation_results ("
        "task_id TEXT, suite_model TEXT, n_base_passing INTEGER, "
        "plus_caught INTEGER, pbt_caught INTEGER, "
        "PRIMARY KEY (task_id, suite_model))"
    )
    plus_by_task = {tid: json.loads(code)["tests"]
                    for tid, code in c.execute("select task_id, code from suites where kind='unit'")}
    tasks = c.execute("select task_id, reference_solution, entry_point, setup, prelude from tasks").fetchall()

    total_overfit_pairs = total_killed = total_plus = 0
    rows = []
    for tid, ref, ep, setup, prelude in tasks:
        if tid not in base:
            continue
        base_tests, base_setup = base[tid]
        plus_tests = plus_by_task.get(tid, [])
        if not plus_tests:
            continue
        # gate: mutants that pass the WEAK base tests (slip past weak public).
        # plus and the PBT are measured comparators over this same set, not gate conditions.
        base_passing = []
        plus_caught = 0
        for m in generate_mutants(ref, limit=limit):
            try:
                if not _passes(base_setup, base_tests, "", m):
                    continue  # caught by weak public; not an overfit candidate
                base_passing.append(m)
                if not _passes(setup or "", plus_tests, prelude or "", m):
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
