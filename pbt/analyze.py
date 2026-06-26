"""Read-only analysis of the results store: the overfit-catch metric.

The research question is whether one model's property-based tests (PBTs) catch
another model's overfitting. An "overfit" program passes the weak unit tests yet
fails a stronger property test, so a PBT that flags it has caught the overfit.

This module reads the denormalized `results` table only. It never writes, and is
meant to run against a connection opened with `pbt.db.connect_ro`, which makes any
mutation physically impossible. The functions accept an already-open connection so
the caller owns its lifecycle.

Definitions:
    eligible: a (program, pbt) pair whose program passes its task's unit suite and
        whose pbt is valid (its task's reference solution passes it).
    caught: an eligible pair whose program fails the pbt (passed = 0).
    overfit-catch rate: caught / eligible over the eligible population.

Validity filter:
    Only pbts marked valid = 1 by pbt.validity count. A pbt whose reference
    solution fails it (valid = 0) or that has not been checked (valid NULL) is a
    suspect property and is excluded from both the eligible and the caught counts,
    so a buggy PBT can never be credited with a false catch. This is enforced by
    joining each pbt result back to its suite on suite_id and requiring
    suites.valid = 1.
"""
import sqlite3

# Pairs eligible for the metric: a program that passes its unit suite, joined to
# every valid pbt result for that program. "caught" marks the pbt that failed
# (passed=0). The join to suites filters out pbts that are invalid or unchecked
# (suites.valid = 1 required), so only trusted properties contribute.
_ELIGIBLE_PAIRS = """
WITH unit_pass AS (
    SELECT DISTINCT program_id
    FROM results
    WHERE kind = 'unit' AND passed = 1
)
SELECT
    r.prog_model                              AS prog_model,
    r.suite_model                             AS suite_model,
    CASE WHEN r.passed = 0 THEN 1 ELSE 0 END  AS caught
FROM results AS r
JOIN unit_pass AS u ON u.program_id = r.program_id
JOIN suites AS s ON s.suite_id = r.suite_id
WHERE r.kind = 'pbt' AND s.valid = 1
"""

_OVERALL_SQL = f"""
SELECT
    COUNT(*)    AS eligible,
    SUM(caught) AS caught
FROM ({_ELIGIBLE_PAIRS})
"""

_MATRIX_SQL = f"""
SELECT
    prog_model,
    suite_model,
    COUNT(*)    AS eligible,
    SUM(caught) AS caught
FROM ({_ELIGIBLE_PAIRS})
GROUP BY prog_model, suite_model
ORDER BY prog_model, suite_model
"""


def _rate(caught: int, eligible: int) -> float | None:
    """Return the catch rate, or None when there is no eligible population.

    Args:
        caught: Number of eligible pairs the pbt caught (program failed the pbt).
        eligible: Number of pairs whose program passes its unit suite.

    Returns:
        caught / eligible as a float, or None if eligible is zero (avoids a
        divide-by-zero on an empty or no-data store).
    """
    return caught / eligible if eligible else None


def overfit_catch_rate(conn: sqlite3.Connection) -> dict:
    """Compute the overall overfit-catch rate across the whole store.

    The eligible population is every (program, pbt) pair whose program passes its
    task's unit suite. A pair is caught when the program fails that pbt.

    Args:
        conn: An open connection to the results store, ideally read-only.

    Returns:
        A dict with keys "eligible" (int), "caught" (int), and "rate"
        (float or None when there are no eligible pairs).
    """
    row = conn.execute(_OVERALL_SQL).fetchone()
    eligible = row["eligible"] or 0
    caught = row["caught"] or 0
    return {"eligible": eligible, "caught": caught, "rate": _rate(caught, eligible)}


def overfit_matrix(conn: sqlite3.Connection) -> list[dict]:
    """Break the overfit-catch rate down by (prog_model, suite_model) cell.

    This is the cross-model view the study asks for: prog_model wrote the program,
    suite_model wrote the pbt, so each cell answers "does suite_model's pbt catch
    prog_model's overfitting?".

    Args:
        conn: An open connection to the results store, ideally read-only.

    Returns:
        One dict per populated cell, each with keys "prog_model", "suite_model",
        "eligible" (int), "caught" (int), and "rate" (float or None). Cells with no
        eligible pairs do not appear. Sorted by prog_model then suite_model.
    """
    cells = []
    for row in conn.execute(_MATRIX_SQL):
        eligible = row["eligible"] or 0
        caught = row["caught"] or 0
        cells.append({
            "prog_model": row["prog_model"],
            "suite_model": row["suite_model"],
            "eligible": eligible,
            "caught": caught,
            "rate": _rate(caught, eligible),
        })
    return cells


def _fmt_rate(rate: float | None) -> str:
    """Format a rate as a percentage string, or a dash when it is None."""
    return "    n/a" if rate is None else f"{rate:7.1%}"


def print_report(conn: sqlite3.Connection) -> None:
    """Print a readable overall summary and cross-model matrix.

    Args:
        conn: An open connection to the results store, ideally read-only.
    """
    overall = overfit_catch_rate(conn)
    print("Overfit-catch rate (program passes unit, pbt catches the failure)")
    print(f"  eligible pairs: {overall['eligible']}")
    print(f"  caught:         {overall['caught']}")
    print(f"  rate:           {_fmt_rate(overall['rate'])}")
    print()
    print(f"{'prog_model':<20} {'suite_model':<20} {'eligible':>9} {'caught':>7} {'rate':>8}")
    print("-" * 68)
    for cell in overfit_matrix(conn):
        print(f"{cell['prog_model']:<20} {cell['suite_model']:<20} "
              f"{cell['eligible']:>9} {cell['caught']:>7} {_fmt_rate(cell['rate']):>8}")
