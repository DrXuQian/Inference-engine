#!/bin/bash
# RAG仓库理解: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
# Input: 800K tokens, Output: 3K tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/06_rag_35B
mkdir -p "$OUT"

echo "=== Split/Prune: RAG仓库理解: Qwen3.5-35B-A3B, TP=1 ==="

python3 "$SCRIPT_DIR/split_and_prune.py" \
    --model-dir "$MODEL" --tp-size 1 \
    --gpu-memory-gb "$GPU_MEM" --max-seq-len 825344 \
    --replicate \
    --output-dir "$OUT/model"

echo "Done: $OUT/model"
