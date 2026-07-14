"""Config-driven orchestrator: seed, generate, validate, evaluate, analyze.

Wires the pipeline end to end for one cross-model cell. One program model writes
candidate code, one suite model writes PBTs, and the run measures how often the
suite model's valid PBTs catch the program model's overfitting (a program that
passes the unit tests but fails a property).

Stages:
    1. seed      load the dataset's tasks and unit suites
    2. generate  sample programs from the program model, PBTs from the suite model
    3. validate  keep only PBTs the reference solution passes
    4. evaluate  score every program against every suite (the cross product)
    5. analyze   report the overfit-catch rate and the cross-model matrix

Generation calls model APIs and costs money; every other stage is local. Pass
model clients to run_pipeline to drive it offline (the tests do this).

Pilot:
    python -m pbt.run --dataset mbppplus --prog-model gpt-4o-mini \\
        --suite-model claude-sonnet-4-5 --n-tasks 10
"""
import argparse
import sqlite3

from pbt import analyze, db, evaluate, validity
from pbt.generate import generate_pbts, generate_programs
from pbt.seed import seed_dataset


def run_pipeline(
    conn: sqlite3.Connection,
    dataset: str,
    prog_model: str,
    suite_model: str,
    n_tasks: int = 10,
    n_samples: int = 1,
    variant: str = "ref",
    prog_client=None,
    suite_client=None,
    difficulties: tuple[str, ...] | None = None,
) -> dict:
    """Run the full pipeline for one cross-model cell and return the metrics.

    Args:
        conn: Open read-write connection to the store.
        dataset: Name of a registered dataset (e.g. "mbppplus", "humaneval").
        prog_model: Model that writes candidate programs.
        suite_model: Model that writes PBTs.
        n_tasks: Number of tasks to seed and run.
        n_samples: Programs sampled per task from the program model.
        variant: PBT prompt variant (ref, noref, nohints, noref_nohints).
        prog_client: Optional stub program model; a real client is built when None.
        suite_client: Optional stub suite model; a real client is built when None.
        difficulties: Optional difficulty tiers to keep (only supported by "apps",
            e.g. ("introductory",) or ("competition",)); None loads the dataset as-is.

    Returns:
        The overall overfit-catch metrics dict, with a "matrix" key holding the
        per (prog_model, suite_model) breakdown.
    """
    n_seeded = seed_dataset(conn, dataset, n_tasks, difficulties=difficulties)
    print(f"[run] seeded {n_seeded} task(s) from {dataset}")

    n_progs = generate_programs(conn, dataset, prog_model, n_samples=n_samples, model=prog_client)
    print(f"[run] generated {n_progs} program(s) with {prog_model}")

    n_pbts = generate_pbts(conn, dataset, suite_model, model=suite_client, variant=variant)
    print(f"[run] generated {n_pbts} PBT(s) with {suite_model}")

    n_checked = validity.validate_pbts(conn, dataset)
    print(f"[run] validated {n_checked} PBT(s) against the reference solution")

    n_pairs = evaluate.evaluate(conn, dataset)
    print(f"[run] scored {n_pairs} new (program, suite) pair(s)")

    overall = analyze.overfit_catch_rate(conn)
    overall["matrix"] = analyze.overfit_matrix(conn)
    return overall


def parse_args() -> argparse.Namespace:
    """Parse the orchestrator's command-line arguments.

    Returns:
        The parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description="Run the cross-model PBT pipeline for one cell.")
    ap.add_argument("--dataset", default="mbppplus")
    ap.add_argument("--prog-model", required=True, help="model that writes candidate programs")
    ap.add_argument("--suite-model", required=True, help="model that writes PBTs")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=1, help="programs sampled per task")
    ap.add_argument("--variant", default="ref",
                    choices=("ref", "noref", "nohints", "noref_nohints"))
    ap.add_argument("--difficulty", default=None,
                    choices=("introductory", "interview", "competition"),
                    help="apps only: keep just this difficulty tier (e.g. introductory)")
    ap.add_argument("--db", default=db.DEFAULT_PATH, help="path to the SQLite store")
    return ap.parse_args()


def main() -> None:
    """Run the pipeline from the command line and print the report."""
    args = parse_args()
    conn = db.connect(args.db)
    try:
        difficulties = (args.difficulty,) if args.difficulty else None
        metrics = run_pipeline(
            conn, args.dataset, args.prog_model, args.suite_model,
            n_tasks=args.n_tasks, n_samples=args.n_samples, variant=args.variant,
            difficulties=difficulties,
        )
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
