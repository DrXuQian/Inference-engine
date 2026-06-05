#!/bin/bash
# Agent batch sweep: Qwen3.5-122B-A10B, TP=1 and TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/04c_agent_batch_122B

echo "=== Split/Prune: Agent Batch Sweep: Qwen3.5-122B-A10B ==="

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
echo "Done: $OUT/"
