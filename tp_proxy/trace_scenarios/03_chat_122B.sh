#!/bin/bash
# Chat问答: Qwen3.5-122B-A10B GPTQ-Int4, TP=1 and TP=2
# Input: 25K, Output: 1K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/03_chat_122B

for TP in 1 2; do
    echo "--- TP=$TP ---"
    MODEL=$(ls -d "$BASE/tp${TP}/model/rank_0_"*L 2>/dev/null | head -1)
    [ -z "$MODEL" ] && echo "Model not found for TP=$TP, run bench_scenario first" && continue
    OUT="$BASE/tp${TP}/trace"

    bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 25600 1024 "$OUT" 10
    echo ""
done
