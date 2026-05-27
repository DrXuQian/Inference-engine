#!/bin/bash
# Agent batch sweep: Qwen3.5-122B-A10B GPTQ-Int4, TP=1 and TP=2
# Tests batch=1,2,4,8 with input=4096, output=1500
# Uses auto_bench.py (same method as other scenarios)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
BASE_04=./results/04_agent_122B
OUT=./results/04c_agent_batch_122B
INPUT_LEN=${INPUT_LEN:-4096}
OUTPUT_LEN=${OUTPUT_LEN:-1500}
NUM_PROMPTS=${NUM_PROMPTS:-20}
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

echo "=== Agent Batch Sweep: Qwen3.5-122B-A10B ==="
echo "Input=$INPUT_LEN, Output=$OUTPUT_LEN, Batch: $BATCH_LIST"

for TP in 1 2; do
    echo ""
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"
    mkdir -p "$TP_DIR"

    # Reuse split model from 04_agent
    PRUNED=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
    if [ -z "$PRUNED" ]; then
        echo "  Split model not found, splitting..."
        python3 "$SCRIPT_DIR/split_and_prune.py" \
            --model-dir "$MODEL" --tp-size $TP \
            --gpu-memory-gb "$GPU_MEM" --max-seq-len $((INPUT_LEN + OUTPUT_LEN + 2048)) \
            --output-dir "$TP_DIR/model"
        PRUNED=$(bash "$SCRIPT_DIR/get_model_path.sh" "$TP_DIR/model")
    fi
    [ -z "$PRUNED" ] && echo "  ERROR: no model" && continue

    for B in $BATCH_LIST; do
        echo ""
        echo "  [TP=$TP batch=$B]"
        python3 "$SCRIPT_DIR/auto_bench.py" \
            --model-dir "$PRUNED" \
            --input-lens $INPUT_LEN \
            --output-len $OUTPUT_LEN \
            --num-prompts $NUM_PROMPTS \
            --batch-size $B \
            --gpu-mem 0.9 \
            --output-json "$TP_DIR/bench_batch${B}.json"
    done
done

echo ""
echo "Done: $OUT/"
