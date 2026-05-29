#!/bin/bash
# Chat问答: Qwen3.5-122B-A10B GPTQ-Int4, TP=1 and TP=2
# Input: 25K, Output: 1K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/03_chat_122B

for TP in 1 2; do
    echo "--- TP=$TP ---"
    MODEL=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/tp${TP}/model" 2>/dev/null || echo "")}
    if [ -z "$MODEL" ]; then
        echo "ERROR: MODEL not set and no split model found for TP=$TP. Either:"
        echo "  1. Set MODEL=/path/to/model env var"
        echo "  2. Run bench_scenarios/03_chat_122B.sh first to split model"
        exit 1
    fi
    OUT="$BASE/tp${TP}/trace"

    bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 25600 1024 "$OUT" 10
    unset MODEL
    echo ""
done
