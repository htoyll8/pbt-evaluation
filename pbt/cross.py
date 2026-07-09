"""Cross-model orchestrator: generate once per model, then evaluate the full product.

Unlike pbt.run (one prog-model x one suite-model cell), this generates each model's
programs and PBTs exactly once, then scores every program against every PBT. For models
{A, B} and task t you get programs P_A, P_B and PBTs Q_A, Q_B, and the evaluation covers
all four pairs (P_A x Q_A, P_A x Q_B, P_B x Q_A, P_B x Q_B). Generating once avoids the
re-sampling that inflates counts when the same model is driven across several cells.

Stages: seed -> generate programs (per model) -> generate PBTs (per model) -> validate
-> evaluate (cross product) -> analyze.

Pilot:
    python -m pbt.cross --dataset mbppplus --models gpt-4o-mini gpt-5.1 --n-tasks 10
"""
import argparse
import sqlite3

from pbt import analyze, db, evaluate, validity
from pbt.generate import generate_pbts, generate_programs
from pbt.seed import seed_dataset


def run_cross(
    conn: sqlite3.Connection,
    dataset: str,
    models: list[str],
    n_tasks: int = 10,
    n_samples: int = 1,
    variant: str = "ref",
    difficulties: tuple[str, ...] | None = None,
) -> dict:
    """Generate programs and PBTs once per model, then evaluate the full cross product.

    Args:
        conn: Open read-write connection to the store.
        dataset: Name of a registered dataset (e.g. "mbppplus", "humaneval").
        models: Models that each write both programs and PBTs.
        n_tasks: Number of tasks to seed and run.
        n_samples: Programs sampled per task from each model.
        variant: PBT prompt variant (ref, noref, nohints, noref_nohints).
        difficulties: Optional difficulty tiers to keep (only supported by "apps",
            e.g. ("introductory",) or ("competition",)); None loads the dataset as-is.

    Returns:
        The overall overfit-catch metrics dict, with a "matrix" key holding the per
        (prog_model, suite_model) breakdown over every model pair.
    """
    n_seeded = seed_dataset(conn, dataset, n_tasks, difficulties=difficulties)
    print(f"[cross] seeded {n_seeded} task(s) from {dataset}")

    for m in models:
        n_progs = generate_programs(conn, dataset, m, n_samples=n_samples)
        print(f"[cross] generated {n_progs} program(s) with {m}")
    for m in models:
        n_pbts = generate_pbts(conn, dataset, m, variant=variant)
        print(f"[cross] generated {n_pbts} PBT(s) with {m}")

    n_checked = validity.validate_pbts(conn, dataset)
    print(f"[cross] validated {n_checked} PBT(s) against the reference solution")

    n_pairs = evaluate.evaluate(conn, dataset)
    print(f"[cross] scored {n_pairs} new (program, suite) pair(s)")

    overall = analyze.overfit_catch_rate(conn)
    overall["matrix"] = analyze.overfit_matrix(conn)
    return overall


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the full cross-model PBT product.")
    ap.add_argument("--dataset", default="mbppplus")
    ap.add_argument("--models", nargs="+", required=True,
                    help="models that each write programs and PBTs (>=2 for a cross)")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=1, help="programs sampled per task per model")
    ap.add_argument("--variant", default="ref",
                    choices=("ref", "noref", "nohints", "noref_nohints"))
    ap.add_argument("--difficulty", default=None,
                    choices=("introductory", "interview", "competition"),
                    help="apps only: keep just this difficulty tier (e.g. introductory)")
    ap.add_argument("--db", default=db.DEFAULT_PATH, help="path to the SQLite store")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    conn = db.connect(args.db)
    try:
        difficulties = (args.difficulty,) if args.difficulty else None
        metrics = run_cross(conn, args.dataset, args.models,
                            n_tasks=args.n_tasks, n_samples=args.n_samples, variant=args.variant,
                            difficulties=difficulties)
    finally:
        conn.close()

    print("\n=== overfit-catch (overall) ===")
    rate = metrics["rate"]
    rate_str = f"{rate:.3f}" if rate is not None else "n/a (no eligible pairs)"
    print(f"  eligible={metrics['eligible']} caught={metrics['caught']} rate={rate_str}")
    print("=== matrix (prog_model x suite_model) ===")
    for cell in metrics["matrix"]:
        cell_rate = cell["rate"]
        cell_str = f"{cell_rate:.3f}" if cell_rate is not None else "n/a"
        print(f"  {cell['prog_model']:24} x {cell['suite_model']:24} "
              f"eligible={cell['eligible']} caught={cell['caught']} rate={cell_str}")


if __name__ == "__main__":
    main()
