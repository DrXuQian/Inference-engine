#!/bin/bash
# Qwen3-30B-A3B GPTQ-Int4, TP=1
# Input: 1.5K tokens, Output: 200/500 tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B-GPTQ-Int4}
OUT=./results/07_qwen3_30b_a3b

if [ ! -d "$MODEL" ]; then
    echo "ERROR: MODEL not found: $MODEL"
    exit 1
fi

echo "=== Qwen3-30B-A3B-GPTQ-Int4, TP=1 ==="

# Trace: 1.5K input, 200 output
echo "--- Output=200 ---"
bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 1536 200 "$OUT/trace_200" 5

# Trace: 1.5K input, 500 output
echo "--- Output=500 ---"
bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 1536 500 "$OUT/trace_500" 5

echo "Done: $OUT/"
