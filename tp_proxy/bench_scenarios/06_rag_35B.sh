#!/bin/bash
# RAG仓库理解: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
# Input: 800K tokens, Output: 3K tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-24}
OUT=./results/06_rag_35B
mkdir -p "$OUT"

echo "=== [6/6] RAG仓库理解: Qwen3.5-35B-A3B, TP=1 ==="
echo "Input=819200, Output=3072"

python3 "$SCRIPT_DIR/split_and_prune.py" \
    --model-dir "$MODEL" --tp-size 1 \
    --gpu-memory-gb "$GPU_MEM" --max-seq-len 825344 \
    --output-dir "$OUT/model"

PRUNED=$(ls -d "$OUT/model/rank_0_"*L 2>/dev/null | head -1)
[ -z "$PRUNED" ] && PRUNED="$MODEL"

python3 "$SCRIPT_DIR/auto_bench.py" \
    --model-dir "$PRUNED" --input-lens 819200 --output-len 3072 \
    --num-prompts 3 --gpu-mem 0.9 \
    --output-json "$OUT/bench.json"

echo "Done: $OUT/bench.json"
