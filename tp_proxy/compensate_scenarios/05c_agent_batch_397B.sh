#!/bin/bash
# Compensate agent batch sweep: per-batch trace and comm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_04=./results/05_agent_397B
OUT=./results/05c_agent_batch_397B
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

for TP in 2; do
    echo "--- TP=$TP ---"
    DIR="$OUT/tp${TP}"
    [ ! -d "$DIR" ] && echo "  Not found" && continue

    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
    [ -z "$MODEL" ] && MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model")
    [ -z "$MODEL" ] && echo "  Model not found" && continue

    for B in $BATCH_LIST; do
        BENCH="$DIR/bench_batch${B}.json"
        [ ! -f "$BENCH" ] && echo "  batch=$B: no bench result" && continue

        echo "  batch=$B"
        TRACE="$DIR/trace_batch${B}/trace.sqlite"
        COMM="$DIR/comm_batch${B}.json"
        COMP_ARGS=""
        [ -f "$TRACE" ] && COMP_ARGS="$COMP_ARGS --asys-sqlite $TRACE"
        [ -f "$COMM" ] && COMP_ARGS="$COMP_ARGS --comm-json $COMM"

        python3 "$SCRIPT_DIR/compensate_ppu.py" \
            --bench-results "$BENCH" \
            --model-dir "$MODEL" \
            $COMP_ARGS \
            --output-json "$DIR/compensated_batch${B}.json"
        echo ""
    done
done
