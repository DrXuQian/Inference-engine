#!/bin/bash
# Qwen3-30B-A3B GPTQ-Int4, TP=2
# Input: 1.5K tokens, Output: 200/500 tokens
# No split needed — use original model directly
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B-GPTQ-Int4}
BASE=./results/07_qwen3_30b_a3b

if [ ! -d "$MODEL" ]; then
    echo "ERROR: MODEL not found: $MODEL"
    exit 1
fi

for TP in 2; do
    echo "=== Qwen3-30B-A3B-GPTQ-Int4, TP=$TP ==="
    DIR="$BASE/tp${TP}"

    for OUTLEN in 200 500; do
        echo "--- TP=$TP, Output=$OUTLEN ---"
        TP_SIZE=$TP bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 1536 "$OUTLEN" "$DIR/trace_${OUTLEN}" 5
    done
    echo ""
done
