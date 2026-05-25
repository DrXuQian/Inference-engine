#!/bin/bash
# Agent长程调用: Qwen 397B-A17B GPTQ-Int4, TP=2 and TP=4
# Input: 100K, Output: 30K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/05_agent_397B

for TP in 2 4; do
    echo "--- TP=$TP ---"
    MODEL=$(ls -d "$BASE/tp${TP}/model/rank_0_"*L 2>/dev/null | head -1)
    [ -z "$MODEL" ] && echo "Model not found for TP=$TP, run bench_scenario first" && continue
    OUT="$BASE/tp${TP}/trace"

    bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 102400 30720 "$OUT" 5
    echo ""
done
