#!/bin/bash
# Agent batch sweep trace: capture trace for each batch size
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_04=./results/04_agent_122B
OUT=./results/04c_agent_batch_122B
INPUT_LEN=${INPUT_LEN:-4096}
OUTPUT_LEN=${OUTPUT_LEN:-1500}
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

for TP in 1 2; do
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"

    PRUNED=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
    [ -z "$PRUNED" ] && PRUNED=$(bash "$SCRIPT_DIR/get_model_path.sh" "$TP_DIR/model")
    [ -z "$PRUNED" ] && echo "  Model not found" && continue

    for B in $BATCH_LIST; do
        echo ""
        echo "  [TP=$TP batch=$B] Capturing trace..."
        TRACE_DIR="$TP_DIR/trace_batch${B}"
        NUM_PROMPTS=$((B * 3 + 2))

        # 6th arg = batch_size (max-concurrency for vllm bench serve)
        bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" $INPUT_LEN $OUTPUT_LEN "$TRACE_DIR" $NUM_PROMPTS $B
    done
done
