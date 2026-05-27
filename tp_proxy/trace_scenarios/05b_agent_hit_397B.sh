#!/bin/bash
# Agent hit: Qwen 397B-A17B, TP=2 and TP=4, input=20K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/05b_agent_hit_397B

for TP in 2; do
    echo "--- TP=$TP ---"
    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/tp${TP}/model")
    [ -z "$MODEL" ] && echo "Model not found for TP=$TP, run bench_scenario first" && continue
    OUT="$BASE/tp${TP}/trace"
    bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 20480 3072 "$OUT" 5
    echo ""
done
