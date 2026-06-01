#!/bin/bash
# Qwen3-30B-A3B GPTQ-Int4, TP=1 and TP=2
# Input: 1.5K tokens, Output: 200/500 tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/07_qwen3_30b_a3b

for TP in 1 2; do
    echo "--- TP=$TP ---"
    MODEL=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/tp${TP}/model" 2>/dev/null || echo "")}
    if [ -z "$MODEL" ]; then
        echo "ERROR: MODEL not set and no split model found for TP=$TP. Either:"
        echo "  1. Set MODEL=/path/to/model env var"
        echo "  2. Run bench_scenarios/07_qwen3_30b_a3b.sh first to split model"
        exit 1
    fi

    DIR="$BASE/tp${TP}"
    for OUTLEN in 200 500; do
        echo "--- TP=$TP, Output=$OUTLEN ---"
        TP_SIZE=$TP bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 1536 "$OUTLEN" "$DIR/trace_${OUTLEN}" 5
    done
    unset MODEL
    echo ""
done
