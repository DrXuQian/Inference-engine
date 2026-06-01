#!/bin/bash
# Qwen3-30B-A3B GPTQ-Int4, TP=2 (MIG: split + run rank_0 on single GPU)
# Input: 1.5K tokens, Output: 200/500 tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/07_qwen3_30b_a3b

for TP in 2; do
    echo "=== Qwen3-30B-A3B-GPTQ-Int4, TP=$TP ==="
    DIR="$BASE/tp${TP}"

    # Use split rank_0 model (from bench_scenarios/07)
    PRUNED=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model" 2>/dev/null || echo "")}
    if [ -z "$PRUNED" ]; then
        echo "ERROR: Split model not found. Run bench_scenarios/07_qwen3_30b_a3b.sh first"
        exit 1
    fi
    echo "Using split model: $PRUNED"

    for OUTLEN in 200 500; do
        echo "--- TP=$TP, Output=$OUTLEN ---"
        # TP_SIZE=1: run on single MIG GPU with split rank_0 weights
        TP_SIZE=1 bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" 1536 "$OUTLEN" "$DIR/trace_${OUTLEN}" 5
    done
    echo ""
done
