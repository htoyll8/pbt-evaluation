# PBT Evaluation

Property-based testing (PBT) generation and evaluation for LLM-generated code.

## Structure

```
scripts/
  generate_pbts.py        # Generate PBTs from task descriptions via LLM
  run_pbt_eval.py         # Evaluate solutions against PBTs
  print_pbt_table.py      # Print summary tables from eval results
  run_pbt_generation.sh   # Bash wrapper for batch generation
model.py                  # LLM API wrapper (Anthropic + OpenAI)
pbt_data/                 # Generated PBTs, eval results, round-robin
results/                  # Seed programs (h0) for MBPP+ and HumanEval
```

## Running evaluation

```bash
# Evaluate Claude seed programs on MBPP+ against Claude-generated PBTs
python scripts/run_pbt_eval.py \
    --results results/claude_mbpp_h0_50.jsonl \
    --pbts pbt_data/pbts_claude_mbppplus_pilot.jsonl \
    --out pbt_data/pbt_eval_output.jsonl \
    --dataset mbppplus

# Print results table
python scripts/print_pbt_table.py pbt_data/pbt_eval_output.jsonl
```

## Generating new PBTs

```bash
python scripts/generate_pbts.py \
    --dataset mbppplus \
    --n 50 \
    --model claude-sonnet-4-5 \
    --variant with-ref \
    --out pbt_data/pbts_new.jsonl
```
