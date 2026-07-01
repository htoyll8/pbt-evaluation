# Handoff: running the cross-model PBT pipeline

Cross-model property-based testing of LLM code: one model writes the program, another writes the
property test (PBT). We measure the **overfit-catch rate** — how often a valid PBT catches a program
that passes the unit tests but is actually wrong.

The current pipeline is the `pbt/` package writing to a **SQLite store** (not the older JSONL
`scripts/` flow the top of the README describes).

## Setup

```bash
git checkout main && git pull     # latest pipeline (merged from feat/mutation-eval)
pip install -r requirements.txt
source ~/.env_keys                # exports OPENROUTER_API_KEY (shared lab key; pilots only)
```

Models route through OpenRouter when `OPENROUTER_API_KEY` is set. **If it is not set, the code
silently falls back to your personal `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and bills you** — for
this project use OpenRouter (the lab key). Verify before a run:

```bash
echo "OpenRouter: $([ -n "$OPENROUTER_API_KEY" ] && echo set || echo NOT set)"
```

Bare names map to slugs (gpt-4o-mini, gpt-5.1, claude-sonnet-4-5, ...); pass an exact slug as
`--prog-model`/`--suite-model` to override. The run prints `Using OpenRouter ...` vs `Using direct
OpenAI ...` so you can confirm the provider.

## Run one cross-model cell (end to end)

```bash
python -m pbt.run \
    --dataset mbppplus \          # mbppplus | humaneval | apps_introductory | apps_competition
    --prog-model gpt-4o-mini \    # writes candidate PROGRAMS
    --suite-model gpt-5.1 \       # writes the PBTs
    --n-tasks 50 \
    --n-samples 5 \               # programs sampled per task
    --db pbt_run.db               # SQLite store
```

Runs five stages and prints the overfit-catch rate plus a per (prog-model x suite-model) matrix:

1. seed     — load tasks + unit suites into the store
2. generate — prog-model writes programs; suite-model writes PBTs  (the only stage that costs money)
3. validate — keep only PBTs the reference solution passes
4. evaluate — score every program x every suite (unit / pbt / private)
5. analyze  — overfit-catch rate + cross-model matrix

Idempotent: re-running with the same `--db` skips work already done (content-hash IDs), so it is safe
to re-run or resume. Only stage 2 hits the API.

## Mutation-catch analysis (optional)

How often a valid PBT catches synthetic overfit — mutants of the canonical solution that pass the
weak public tests. Writes a `mutation_results` table to the same db.

```bash
PYTHONPATH=. python scripts/mutation_eval.py pbt_run.db --dataset mbppplus
```

## Where results live

- SQLite store (`--db`): tables `tasks, programs, generations, suites, results` (+ `mutation_results`).
  Read-only inspect: `sqlite3 'file:pbt_run.db?mode=ro'`.
- MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (set `PBT_MLFLOW=0` to disable).

## Notes

- The PBT prompt lives in `scripts/generate_pbts.py` (`PBT_PROMPT`); reused by `pbt/generate.py`.
- Recent change: idempotency is only suggested when the operation is genuinely idempotent (it drove
  most invalid PBTs otherwise).
- Do not commit `*.db`, `pbt_data/`, `results/`, or API keys (see `.gitignore`).
