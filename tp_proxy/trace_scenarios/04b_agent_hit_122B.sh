#!/bin/bash
# Agent hit: Qwen3.5-122B-A10B, TP=1 and TP=2, input=20K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/04b_agent_hit_122B

for TP in 1 2; do
    echo "--- TP=$TP ---"
    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/tp${TP}/model")
    [ -z "$MODEL" ] && echo "Model not found for TP=$TP, run bench_scenario first" && continue
    OUT="$BASE/tp${TP}/trace"
    bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 20480 3072 "$OUT" 5
    echo ""
done
