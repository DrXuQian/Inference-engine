#!/bin/bash
# Agent batch sweep: Qwen3.5-397B-A17B, TP=1 and TP=2
# Uses offline mode (vllm.LLM) for guaranteed proper batching
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
BASE_04=./results/05_agent_397B
OUT=./results/05c_agent_batch_397B
INPUT_LEN=${INPUT_LEN:-102400}
OUTPUT_LEN=${OUTPUT_LEN:-3072}
NUM_PROMPTS=${NUM_PROMPTS:-40}
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

echo "=== Agent Batch Sweep (offline mode): Qwen3.5-397B-A17B ==="
echo "Input=$INPUT_LEN, Output=$OUTPUT_LEN, Batch: $BATCH_LIST"

for TP in 2; do
    echo ""
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"
    mkdir -p "$TP_DIR"

    PRUNED=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
    if [ -z "$PRUNED" ]; then
        echo "  Split model not found, splitting..."
        python3 "$SCRIPT_DIR/split_and_prune.py" \
            --model-dir "$MODEL" --tp-size $TP \
            --gpu-memory-gb "$GPU_MEM" --max-seq-len 108544 \
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
            --mode offline \
            --gpu-mem 0.9 \
            --output-json "$TP_DIR/bench_batch${B}.json"
    done
done

echo ""
echo "Done: $OUT/"
