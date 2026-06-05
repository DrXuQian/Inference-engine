#!/bin/bash
# Capture trace for each batch size (lm_head kernel changes with batch)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_04=./results/04_agent_122B
OUT=./results/04c_agent_batch_122B
INPUT_LEN=${INPUT_LEN:-102400}
OUTPUT_LEN=${OUTPUT_LEN:-3072}
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

for TP in 1 2; do
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"

    PRUNED=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model" 2>/dev/null || echo "")}
    if [ -z "$PRUNED" ]; then
        echo "ERROR: MODEL not set and no split model found for TP=$TP. Either:"
        echo "  1. Set MODEL=/path/to/model env var"
        echo "  2. Run bench_scenarios/04_agent_122B.sh first to split model"
        exit 1
    fi

    for B in $BATCH_LIST; do
        echo ""
        echo "  [TP=$TP batch=$B] Capturing trace..."
        TRACE_DIR="$TP_DIR/trace_batch${B}"
        NUM_PROMPTS=$((B * 3 + 2))
        # 6th arg = batch_size
        bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" $INPUT_LEN $OUTPUT_LEN "$TRACE_DIR" $NUM_PROMPTS $B
    done
done
