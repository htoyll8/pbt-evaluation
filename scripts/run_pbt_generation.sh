#!/bin/bash
# Generate PBTs for HumanEval (foreground, ~20 tasks) and MBPP+ (background tmux, 50 tasks).
# Usage: bash scripts/run_pbt_generation.sh

set -u

DIR="$(cd "$(dirname "$0")/.." && pwd)"

source ~/.env_keys
export ANTHROPIC_VERTEX_PROJECT=your-gcp-project
export ANTHROPIC_VERTEX_REGION=us-east5

# HumanEval — run in foreground (fast, ~20 tasks)
echo "==> Generating HumanEval PBTs (foreground)..."
python3 "$DIR/scripts/generate_pbts.py" \
    --dataset humaneval \
    --model claude-sonnet-4-5 \
    --max_tasks 20 \
    --out "$DIR/pbt_data/pbts_claude_humaneval_pilot.jsonl"

# MBPP+ — run overnight in tmux
echo ""
echo "==> Launching MBPP+ PBT generation in tmux session 'pbt_gen'..."
tmux kill-session -t pbt_gen 2>/dev/null || true
tmux new-session -d -s pbt_gen \
    "source ~/.env_keys && \
     export ANTHROPIC_VERTEX_PROJECT=your-gcp-project && \
     export ANTHROPIC_VERTEX_REGION=us-east5 && \
     python3 $DIR/scripts/generate_pbts.py \
       --dataset mbppplus \
       --model claude-sonnet-4-5 \
       --max_tasks 50 \
       --out $DIR/pbt_data/pbts_claude_mbppplus_pilot.jsonl \
       > $DIR/logs/pbt_generation_mbpp.log 2>&1 && \
     echo 'MBPP+ PBT generation done'"

echo "MBPP+ running in background. Monitor with:"
echo "  tail -f $DIR/logs/pbt_generation_mbpp.log"
