#!/bin/bash
# Chat问答: 27B FP16, TP=1
# Input: 25K tokens, Output: 1K tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-27B}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/02_chat_27B
mkdir -p "$OUT"

echo "=== [2/6] Chat问答: 27B FP16, TP=1 ==="
echo "Input=25600, Output=1024"

python3 "$SCRIPT_DIR/split_and_prune.py" \
    --model-dir "$MODEL" --tp-size 1 \
    --gpu-memory-gb "$GPU_MEM" --max-seq-len 27648 \
    --output-dir "$OUT/model"

PRUNED=$(python3 -c "import json; print(json.load(open('$OUT/model/split_meta.json'))[\"output_dir\"])" 2>/dev/null)
[ -z "$PRUNED" ] && PRUNED="$MODEL"

python3 "$SCRIPT_DIR/auto_bench.py" \
    --model-dir "$PRUNED" --input-lens 25600 --output-len 1024 \
    --num-prompts 10 --gpu-mem 0.9 \
    --output-json "$OUT/bench.json"

echo "Done: $OUT/bench.json"
