#!/bin/bash
# Agent长程调用: Qwen3.5-122B-A10B GPTQ-Int4, TP=1 and TP=2
# Input: 100K tokens, Output: 3K tokens
# Full task: 1M = first 100K (0% hit) + 9 × 100K (80% hit)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/04_agent_122B
mkdir -p "$OUT"

echo "=== Split/Prune: Agent长程调用: Qwen3.5-122B-A10B, TP=1&2 ==="

for TP in 1 2; do
    echo ""
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"
    mkdir -p "$TP_DIR"

    python3 "$SCRIPT_DIR/split_and_prune.py" \
        --model-dir "$MODEL" --tp-size $TP \
        --gpu-memory-gb "$GPU_MEM" --max-seq-len 108544 \
        --replicate \
        --output-dir "$TP_DIR/model"
done

echo ""
echo "Done: $OUT/tp1/model, $OUT/tp2/model"
