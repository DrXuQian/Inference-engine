#!/bin/bash
# 代码补全: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
# Input: 15K tokens, Output: 50 tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/01_code_completion_35B
mkdir -p "$OUT"

echo "=== [1/6] 代码补全: Qwen3.5-35B-A3B, TP=1 ==="
echo "Input=15360, Output=50"

python3 "$SCRIPT_DIR/split_and_prune.py" \
    --model-dir "$MODEL" --tp-size 1 \
    --gpu-memory-gb "$GPU_MEM" --max-seq-len 16384 \
    --output-dir "$OUT/model"

PRUNED=$(ls -d "$OUT/model/rank_0_"*L 2>/dev/null | head -1)
[ -z "$PRUNED" ] && PRUNED="$MODEL"

python3 "$SCRIPT_DIR/auto_bench.py" \
    --model-dir "$PRUNED" --input-lens 15360 --output-len 50 \
    --num-prompts 10 --gpu-mem 0.9 \
    --output-json "$OUT/bench.json"

echo "Done: $OUT/bench.json"
