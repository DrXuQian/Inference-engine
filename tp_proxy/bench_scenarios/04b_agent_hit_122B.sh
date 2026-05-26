#!/bin/bash
# Agent长程调用(cache hit): Qwen3.5-122B-A10B GPTQ-Int4, TP=1 and TP=2
# Input: 20K tokens (80% prefix cache hit scenario), Output: 3K tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
OUT=./results/04b_agent_hit_122B
mkdir -p "$OUT"

echo "=== Agent cache hit: Qwen3.5-122B-A10B, TP=1&2 ==="
echo "Input=20480, Output=3072"

for TP in 1 2; do
    echo ""
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"
    mkdir -p "$TP_DIR"

    python3 "$SCRIPT_DIR/split_and_prune.py" \
        --model-dir "$MODEL" --tp-size $TP \
        --gpu-memory-gb "$GPU_MEM" --max-seq-len 26624 \
        --output-dir "$TP_DIR/model"

    PRUNED=$(ls -d "$TP_DIR/model/rank_0_"*L 2>/dev/null | head -1)
    [ -z "$PRUNED" ] && PRUNED=$(ls -d "$TP_DIR/model/split/rank_0" 2>/dev/null | head -1)
    [ -z "$PRUNED" ] && PRUNED="$MODEL"

    python3 "$SCRIPT_DIR/auto_bench.py" \
        --model-dir "$PRUNED" --input-lens 20480 --output-len 3072 \
        --num-prompts 5 --gpu-mem 0.9 \
        --output-json "$TP_DIR/bench.json"
done

echo ""
echo "Done: $OUT/tp1/bench.json, $OUT/tp2/bench.json"
