#!/bin/bash
# Qwen3-30B-A3B BF16, TP=1/2/4 (MIG/proxy: split + run rank_0 on single GPU)
# Input: 1.5K tokens, Output: 200/500 tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/07_qwen3_30b_a3b
TPS="${TPS:-1 2 4}"
OUTLENS="${OUTLENS:-200 500}"
CAPTURE_ITERS="${CAPTURE_ITERS:-5}"

for TP in $TPS; do
    echo "=== Qwen3-30B-A3B-BF16, TP=$TP ==="
    DIR="$BASE/tp${TP}"

    # Use split/pruned rank_0 model (from bench_scenarios/07).
    PRUNED=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model" 2>/dev/null || echo "")}
    if [ -z "$PRUNED" ]; then
        echo "ERROR: Split model not found for TP=$TP. Run bench_scenarios/07_qwen3_30b_a3b.sh first"
        exit 1
    fi
    echo "Using split model: $PRUNED"

    for OUTLEN in $OUTLENS; do
        echo "--- TP=$TP, Output=$OUTLEN ---"
        # TP_SIZE=1: run on one GPU with split/pruned rank_0 weights.
        TP_SIZE=1 bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" 1536 "$OUTLEN" "$DIR/trace_${OUTLEN}" "$CAPTURE_ITERS"
    done
    echo ""
done
