#!/bin/bash
# Agent长程调用: Qwen 397B-A17B GPTQ-Int4, TP=2 and TP=4
# Input: 100K, Output: 3K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/05_agent_397B

for TP in 2; do
    echo "--- TP=$TP ---"
    MODEL=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/tp${TP}/model" 2>/dev/null || echo "")}
    if [ -z "$MODEL" ]; then
        echo "ERROR: MODEL not set and no split model found for TP=$TP. Either:"
        echo "  1. Set MODEL=/path/to/model env var"
        echo "  2. Run bench_scenarios/05_agent_397B.sh first to split model"
        exit 1
    fi
    OUT="$BASE/tp${TP}/trace"

    bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 102400 3072 "$OUT" 5
    unset MODEL
    echo ""
done
