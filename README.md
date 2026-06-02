# Can LLMs Test Their Own Code? Cross-Model Property-Based Testing of Code

**Tyler Holloway**, Youhui Wang, Simon Henniger, Nada Amin · Harvard
*ICML 2026 FAGEN Workshop · arXiv preprint and citation coming soon*

Evaluation harness for cross-model property-based testing of LLM-generated code: one model writes the code, another writes the property-based test (PBT). PBTs catch failures the unit tests miss.

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

### 1. Generate PBTs

```bash
python scripts/generate_pbts.py \
    --dataset mbppplus \             # mbppplus | humaneval | apps_introductory | apps_competition
    --model claude-sonnet-4-5 \      # any Anthropic or OpenAI model ID
    --max_tasks 50                   # tasks to process
```

Output is auto-named from the model, dataset, and ablation flags: `pbt_data/pbts_<MODEL>_<DATASET>_<VARIANT>.jsonl`. Override with `--out`.

Ablation flags: `--no-ref` (omits reference), `--two-step` (extract properties first), `--no-hints` (omits property hints). Run with `--help` for details.

### 2. Evaluate seeds against PBTs

```bash
python scripts/run_pbt_eval.py \
    --results results/claude-sonnet-4-5_mbppplus_h0_50.jsonl \         # seed programs from a code-generating LLM
    --pbts pbt_data/pbts_claude-sonnet-4-5_mbppplus_withref.jsonl \    # PBTs from step 1
    --out pbt_data/pbt_eval_output.jsonl \                             # eval results; feeds step 3
    --dataset mbppplus                                                  # same benchmark as the seeds
```

`--timeout` (default 30s) bounds each PBT run. Run with `--help` for the full list.

### 3. Print summary table

```bash
python scripts/print_pbt_table.py pbt_data/pbt_eval_output.jsonl
```

## File naming convention

Files follow a self-describing pattern:

| Pattern | Example |
|---|---|
| Seed programs | `results/<MODEL>_<DATASET>_h0_<N>.jsonl` |
| Generated PBTs | `pbt_data/pbts_<MODEL>_<DATASET>_<VARIANT>.jsonl` |
| Eval results | `pbt_data/pbt_eval_<MODEL>_<DATASET>_<VARIANT>.jsonl` |

`<VARIANT>` encodes the ablation flags: `withref`, `noref`, `withref_twostep`, `noref_nohints`, etc.

## Repository structure

```
pbt-evaluation/
├── scripts/
│   ├── generate_pbts.py        # Generate PBTs from task descriptions
│   ├── run_pbt_eval.py         # Evaluate solutions against PBTs
│   ├── print_pbt_table.py      # Summary tables
│   └── run_pbt_generation.sh   # Batch wrapper
├── model.py                    # LLM API wrapper (Anthropic + OpenAI)
├── pbt_data/                   # Generated PBTs, eval results, round-robin runs
├── results/                    # Seed programs (h0) for MBPP+ and HumanEval
├── pbt_plan.md                 # Design notes
├── pbt_results.md              # Results notebook
└── fig_pbt_pipeline.tex        # Pipeline diagram source
```

## Setup

- **Code generators:** Claude Sonnet 4.5, GPT-4, GPT-5.1
- **PBT generators:** Claude Sonnet 4.5, GPT-5.1
- **Benchmarks:** MBPP+, HumanEval

## License

MIT.

## Contact

Tyler Holloway · `tylerholloway@g.harvard.edu`
