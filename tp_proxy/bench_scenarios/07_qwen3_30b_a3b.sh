#!/bin/bash
# Qwen3-30B-A3B GPTQ-Int4, TP=1 and TP=2
# Input: 1.5K tokens, max-seq-len: 2048
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/07_qwen3_30b_a3b
mkdir -p "$OUT"

echo "=== Split/Prune: Qwen3-30B-A3B-GPTQ-Int4, TP=1&2 ==="

for TP in 1 2; do
    echo ""
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"
    mkdir -p "$TP_DIR"

    python3 "$SCRIPT_DIR/split_and_prune.py" \
        --model-dir "$MODEL" --tp-size $TP \
        --gpu-memory-gb "$GPU_MEM" --max-seq-len 2048 \
        --output-dir "$TP_DIR/model"
done

echo ""
echo "Done: $OUT/tp1/model, $OUT/tp2/model"
