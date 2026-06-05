#!/bin/bash
# Agent长程调用(cache hit): Qwen 397B-A17B GPTQ-Int4, TP=2 and TP=4
# Input: 20K tokens (80% prefix cache hit scenario), Output: 3K tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/05b_agent_hit_397B
mkdir -p "$OUT"

echo "=== Split/Prune: Agent cache hit: Qwen 397B-A17B, TP=2&4 ==="

for TP in 2 4; do
    echo ""
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"
    mkdir -p "$TP_DIR"

    python3 "$SCRIPT_DIR/split_and_prune.py" \
        --model-dir "$MODEL" --tp-size $TP \
        --gpu-memory-gb "$GPU_MEM" --max-seq-len 26624 \
        --output-dir "$TP_DIR/model"
done

echo ""
echo "Done: $OUT/tp2/model  $OUT/tp4/model"
