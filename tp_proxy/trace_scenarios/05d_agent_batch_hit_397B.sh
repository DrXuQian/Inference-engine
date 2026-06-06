#!/bin/bash
# Capture trace for batch hit sweep (20K input, batch=1,2,4,8)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_05=./results/05_agent_397B
OUT=./results/05d_agent_batch_hit_397B
INPUT_LEN=${INPUT_LEN:-20480}
OUTPUT_LEN=${OUTPUT_LEN:-3072}
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

for TP in 2 4; do
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"

    PRUNED=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_05/tp${TP}/model" 2>/dev/null || echo "")}
    if [ -z "$PRUNED" ]; then
        echo "ERROR: MODEL not set and no split model found for TP=$TP. Either:"
        echo "  1. Set MODEL=/path/to/model env var"
        echo "  2. Run bench_scenarios/05_agent_397B.sh first to split model"
        exit 1
    fi

    for B in $BATCH_LIST; do
        echo ""
        echo "  [TP=$TP batch=$B] Capturing trace..."
        TRACE_DIR="$TP_DIR/trace_batch${B}"
        NUM_PROMPTS=$((B * 3 + 2))
        bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" $INPUT_LEN $OUTPUT_LEN "$TRACE_DIR" $NUM_PROMPTS $B
    done
done
