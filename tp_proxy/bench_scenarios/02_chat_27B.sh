#!/bin/bash
# Chat问答: 27B FP16, TP=1
# Input: 25K tokens, Output: 1K tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-27B}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/02_chat_27B
mkdir -p "$OUT"

echo "=== Split/Prune: Chat问答: 27B FP16, TP=1 ==="

python3 "$SCRIPT_DIR/split_and_prune.py" \
    --model-dir "$MODEL" --tp-size 1 \
    --gpu-memory-gb "$GPU_MEM" --max-seq-len 27648 \
    --replicate \
    --output-dir "$OUT/model"

echo "Done: $OUT/model"
