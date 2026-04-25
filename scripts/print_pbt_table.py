"""
print_pbt_table.py

Prints the PBT results summary table from pbt_eval JSONL files.
Seeds only (attempt=0).

Usage:
    python scripts/print_pbt_table.py
"""

import json
from collections import defaultdict
from pathlib import Path

MBPP_FILES = {
    "Claude":   "pbt_data/pbt_eval_claude_mbppplus_new.jsonl",
    "GPT-5.1":  "pbt_data/pbt_eval_gpt51_mbppplus_pilot.jsonl",
}

HUMANEVAL_FILES = {
    "Claude":   "pbt_data/pbt_eval_claude_humaneval_pilot.jsonl",
    "GPT-5.1":  "pbt_data/pbt_eval_gpt51_humaneval_pilot.jsonl",
}

# Private eval files for task-level any/all stats (120 tasks)
PRIVATE_EVAL_FILES = {
    "Claude":   None,  # not yet generated for Claude
    "GPT-5.1":  "pbt_data/private_eval_gpt51_mbppplus.jsonl",
    "GPT-4":    "pbt_data/private_eval_gpt4_mbppplus.jsonl",
}


def load_seeds(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("attempt", 0) == 0:
                rows.append(r)
    return rows


def task_level_stats(private_eval_path: str) -> dict:
    """Compute any/all seed passes private, from a private_eval JSONL."""
    if not private_eval_path or not Path(private_eval_path).exists():
        return {}
    rows = [json.loads(l) for l in open(private_eval_path)]
    task_seeds = defaultdict(list)
    for r in rows:
        task_seeds[r["task_id"]].append(r)
    n_tasks = len(task_seeds)
    any_pass = sum(1 for seeds in task_seeds.values() if any(r["passes_private"] for r in seeds))
    all_pass = sum(1 for seeds in task_seeds.values() if all(r["passes_private"] for r in seeds))
    return {
        "n_tasks": n_tasks,
        "any_pass": any_pass,
        "all_pass": all_pass,
    }


def stats(rows: list, has_private: bool) -> dict:
    total = len(rows)
    unit_pass = [r for r in rows if r.get("passes_unit")]
    pbt_pass  = [r for r in rows if r.get("passes_pbt")]
    overfit_pbt = [r for r in rows if r.get("overfit_pbt")]

    d = {
        "total":        total,
        "unit_pass":    len(unit_pass),
        "pbt_pass":     len(pbt_pass),
        "overfit_pbt":  len(overfit_pbt),
        "overfit_pbt_rate": len(overfit_pbt) / len(unit_pass) * 100 if unit_pass else None,
    }

    if has_private:
        priv_pass    = [r for r in rows if r.get("passes_private")]
        overfit_priv = [r for r in rows if r.get("overfit_private")]
        overfit_both = [r for r in rows if r.get("overfit_pbt") and r.get("overfit_private")]
        d["priv_pass"]        = len(priv_pass)
        d["overfit_priv"]     = len(overfit_priv)
        d["overfit_priv_rate"] = len(overfit_priv) / len(unit_pass) * 100 if unit_pass else None
        d["overfit_both"]     = len(overfit_both)
        d["overfit_both_rate"] = len(overfit_both) / len(unit_pass) * 100 if unit_pass else None

    return d


def pct(n, total):
    if total == 0:
        return "—"
    return f"{n} ({n/total*100:.1f}%)"


def rate(v):
    if v is None:
        return "N/A"
    return f"{v:.1f}%"


def print_table(title, file_map, private_map=None):
    columns = {}
    for label, path in file_map.items():
        if not Path(path).exists():
            columns[label] = None
            continue
        rows = load_seeds(path)
        has_private = any("passes_private" in r for r in rows)
        columns[label] = stats(rows, has_private)

    task_cols = {}
    if private_map:
        for label, path in private_map.items():
            task_cols[label] = task_level_stats(path)

    col_names = list(file_map.keys())
    col_width = 22

    def row(label, values):
        cells = [label.ljust(30)] + [str(v).ljust(col_width) for v in values]
        print("  ".join(cells))

    def divider():
        print("-" * (32 + (col_width + 2) * len(col_names)))

    # Header
    print()
    print("  ".join(["".ljust(30)] + [c.ljust(col_width) for c in col_names]))
    divider()

    def get(col, key, fmt="raw"):
        s = columns.get(col)
        if s is None:
            return "—"
        v = s.get(key)
        if v is None:
            return "N/A"
        if fmt == "pct":
            return pct(v, s["total"])
        if fmt == "rate":
            return rate(v)
        return str(v)

    row("Seeds evaluated",          [get(c, "total") for c in col_names])
    row("Passes unit tests",        [get(c, "unit_pass", "pct") for c in col_names])
    row("Passes PBT",               [get(c, "pbt_pass", "pct") for c in col_names])
    row("Passes private harness",   [
        get(c, "priv_pass", "pct") if columns.get(c) and "priv_pass" in (columns[c] or {}) else "N/A"
        for c in col_names
    ])
    divider()
    row("Overfit rate (PBT)",       [get(c, "overfit_pbt_rate", "rate") for c in col_names])
    row("Overfit rate (private)",   [
        get(c, "overfit_priv_rate", "rate") if columns.get(c) and "overfit_priv_rate" in (columns[c] or {}) else "N/A"
        for c in col_names
    ])
    row("Overfit rate (both)",      [
        get(c, "overfit_both_rate", "rate") if columns.get(c) and "overfit_both_rate" in (columns[c] or {}) else "N/A"
        for c in col_names
    ])

    if task_cols:
        divider()
        def task_pct(col, key):
            t = task_cols.get(col, {})
            if not t:
                return "—"
            n = t.get(key, 0)
            total = t.get("n_tasks", 0)
            return pct(n, total)

        row("Task-level: any seed passes", [task_pct(c, "any_pass") for c in col_names])
        row("Task-level: all seeds pass",  [task_pct(c, "all_pass") for c in col_names])

    print()
    print(f"Note: overfit rates are % of unit-passing seeds. Seeds only (attempt=0).")
    print()


def main():
    print("\n=== MBPP+ ===")
    print_table("MBPP+", MBPP_FILES, private_map=PRIVATE_EVAL_FILES)
    print("\n=== HumanEval ===")
    print_table("HumanEval", HUMANEVAL_FILES)


if __name__ == "__main__":
    main()
