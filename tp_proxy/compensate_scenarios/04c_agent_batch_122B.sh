#!/bin/bash
# Compensate agent batch sweep results
# Applies layer-scale compensation to each batch level's TPOT
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/04c_agent_batch_122B
TRACE_BASE=./results/04_agent_122B

for TP in 1 2; do
    echo "--- TP=$TP ---"
    DIR="$BASE/tp${TP}"
    [ ! -d "$DIR" ] && echo "  Not found" && continue

    # Use trace from 04_agent (tail is batch-independent)
    TRACE="$TRACE_BASE/tp${TP}/trace/trace.sqlite"
    COMM="$TRACE_BASE/tp${TP}/comm.json"
    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$TRACE_BASE/tp${TP}/model")
    [ -z "$MODEL" ] && MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model")
    [ -z "$MODEL" ] && echo "  Model not found" && continue

    for LOG in "$DIR"/batch*.log; do
        [ ! -f "$LOG" ] && continue
        B=$(basename "$LOG" .log | sed 's/batch//')
        echo "  batch=$B"

        # Extract TPOT/TTFT from log
        TPOT=$(grep -i "median.*tpot\|median.*inter-token" "$LOG" | tail -1 | awk -F: '{print $NF}' | tr -d ' ')
        TTFT=$(grep -i "median.*ttft" "$LOG" | tail -1 | awk -F: '{print $NF}' | tr -d ' ')

        [ -z "$TPOT" ] && echo "    No TPOT found" && continue

        echo "    raw TPOT=${TPOT}ms, TTFT=${TTFT}ms"

        # Create a minimal bench.json for compensate_ppu.py
        python3 -c "
import json
r = {'input_len': 4096, 'output_len': 1500, 'output_tokens': 1500,
     'tpot_median_ms': $TPOT, 'ttft_median_ms': ${TTFT:-0}}
json.dump({'results': [r]}, open('$DIR/bench_batch${B}.json', 'w'), indent=2)
"
        COMP_ARGS=""
        [ -f "$TRACE" ] && COMP_ARGS="$COMP_ARGS --asys-sqlite $TRACE"
        [ -f "$COMM" ] && COMP_ARGS="$COMP_ARGS --comm-json $COMM"

        python3 "$SCRIPT_DIR/compensate_ppu.py" \
            --bench-results "$DIR/bench_batch${B}.json" \
            --model-dir "$MODEL" \
            $COMP_ARGS \
            --output-json "$DIR/compensated_batch${B}.json"
        echo ""
    done
done
