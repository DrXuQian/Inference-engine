#!/bin/bash
# Agent batch sweep: Qwen3.5-397B-A17B, TP=1 and TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/05c_agent_batch_397B

echo "=== Split/Prune: Agent Batch Sweep: Qwen3.5-397B-A17B ==="

for TP in 2 4; do
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
