"""Mutation-based overfit-catch eval.

For each task we mutate the canonical solution and keep the mutants that slip past the WEAK
public tests (the gate): synthetic overfit programs that look correct under the public suite.
Over that gated set we then measure two comparators, the STRONG private tests (plus) and each
valid PBT. The headline number is the PBT catch rate (pbt_caught / base_passing); plus is a
reference baseline, not a gate condition.

Weak gate and strong comparator by dataset:
  mbppplus  : weak = base `test_list`; strong = the EvalPlus plus harness.
  humaneval : weak = original HumanEval `check()` asserts; strong = the HumanEval+ harness.

The weak gate is fetched fresh from the benchmark (see load_base_tests). The strong comparator
is read from the store's kind='private' suites, which pbt.seed writes for both datasets (see
load_plus_tests). Stores seeded before the weak-gate migration have no private suites and are
rejected: back then mbppplus seeded the plus harness as kind='unit', which is now the weak gate.

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


def load_plus_tests(conn: sqlite3.Connection) -> dict:
    """task_id -> (tests, prelude, per_timeout): the STRONG comparator, from the store.

    The strong comparator is the benchmark's augmented harness, which pbt.seed now writes as a
    kind='private' suite for every dataset whose loader supplies `Task.private_tests`: MBPP+'s
    plus harness, HumanEval's humanevalplus. Reading it from the store rather than re-fetching
    keeps this comparator byte-identical to the one pbt.evaluate scores, so the mutation track
    and the main pipeline cannot drift apart.

    `prelude` and `per_timeout` are taken from the suite blob when it carries them (MBPP+'s plus
    cases need the harness preamble bound; HumanEval's harness runs whole and needs a longer
    timeout), and are None otherwise, meaning "inherit the task's".

    NB: this previously fell back to kind='unit' for mbppplus, back when that loader seeded the
    plus harness as the unit suite. kind='unit' is now the WEAK gate, so reading it here would
    make the strong comparator identical to the gate and force plus_caught to 0.
    """
    out = {}
    for tid, code in conn.execute("select task_id, code from suites where kind='private'"):
        blob = json.loads(code)
        out[str(tid)] = (blob["tests"], blob.get("prelude"), blob.get("per_timeout"))
    return out


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
    # Strong comparator: the seeded kind='private' suite (the benchmark's augmented harness),
    # carrying its own prelude/timeout where it needs one; see load_plus_tests.
    plus_by_task = load_plus_tests(c)
    if not plus_by_task:
        raise SystemExit(
            f"no kind='private' suites in {db!r}: the strong comparator has nowhere to come from. "
            "This store predates the weak-gate migration; rebuild it with "
            "scripts/migrate_weak_gate.py."
        )
    tasks = c.execute("select task_id, reference_solution, entry_point, setup, prelude from tasks").fetchall()

    total_overfit_pairs = total_killed = total_plus = 0
    rows = []
    for tid, ref, ep, setup, prelude in tasks:
        if tid not in base:
            continue
        base_tests, base_setup, base_prelude = base[tid]
        entry = plus_by_task.get(tid)
        if not entry:
            continue
        plus_tests, plus_prelude, plus_timeout = entry
        if plus_prelude is None:
            plus_prelude = prelude or ""
        if plus_timeout is None:
            plus_timeout = 5
        # gate: mutants that pass the WEAK base tests (slip past weak public).
        # plus and the PBT are measured comparators over this same set, not gate conditions.
        base_passing = []
        plus_caught = 0
        for m in generate_mutants(ref, limit=limit):
            try:
                if not _passes(base_setup, base_tests, base_prelude, m):
                    continue  # caught by weak public; not an overfit candidate
                base_passing.append(m)
                if not _passes(setup or "", plus_tests, plus_prelude, m, per_timeout=plus_timeout):
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
