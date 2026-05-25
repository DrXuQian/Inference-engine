#!/bin/bash
# Agent长程调用: Qwen 397B-A17B GPTQ-Int4, TP=2 and TP=4
# Input: 100K tokens, Output: 30K tokens
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-397B-A27B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-24}
OUT=./results/05_agent_397B
mkdir -p "$OUT"

echo "=== [5/6] Agent长程调用: Qwen 397B-A17B, TP=2&4 ==="
echo "Input=102400, Output=30720"

for TP in 2 4; do
    echo ""
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"
    mkdir -p "$TP_DIR"

    python3 "$SCRIPT_DIR/split_and_prune.py" \
        --model-dir "$MODEL" --tp-size $TP \
        --gpu-memory-gb "$GPU_MEM" --max-seq-len 135168 \
        --output-dir "$TP_DIR/model"

    PRUNED=$(ls -d "$TP_DIR/model/rank_0_"*L 2>/dev/null | head -1)
    [ -z "$PRUNED" ] && PRUNED=$(ls -d "$TP_DIR/model/split/rank_0" 2>/dev/null | head -1)
    [ -z "$PRUNED" ] && PRUNED="$MODEL"

    python3 "$SCRIPT_DIR/auto_bench.py" \
        --model-dir "$PRUNED" --input-lens 102400 --output-len 30720 \
        --num-prompts 5 --gpu-mem 0.9 \
        --output-json "$TP_DIR/bench.json"
done

echo ""
echo "Done: $OUT/tp2/bench.json, $OUT/tp4/bench.json"
