#!/bin/bash
# Compensate agent batch hit sweep (20K input, actual KV=100K)
# Uses per-batch trace: batch=1 uses gemvt_op, batch>1 uses gemm kernel
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_05=./results/05_agent_397B
BASE_05B=./results/05b_agent_hit_397B
OUT=./results/05d_agent_batch_hit_397B
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"
LM_HEAD_B1="gemvt_op"
LM_HEAD_BN="gemm_ktype0_aiu1_mtype1_dtypeBF16xBF16xFP32xBF16xBF16"

for TP in 2 4; do
    echo "--- TP=$TP ---"
    DIR="$OUT/tp${TP}"
    [ ! -d "$DIR" ] && echo "  Not found" && continue

    MODEL_TP=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_05/tp${TP}/model" 2>/dev/null || echo "")}
    [ -z "$MODEL_TP" ] && MODEL_TP=$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model" 2>/dev/null || echo "")
    if [ -z "$MODEL_TP" ]; then
        echo "  ERROR: Model not found for TP=$TP, skipping"
        continue
    fi

    COMM="$BASE_05/tp${TP}/comm.json"; [ -f "$COMM" ] && COMM_ARG="--comm-json $COMM" || COMM_ARG=""

    for B in $BATCH_LIST; do
        TRACE="$DIR/trace_batch${B}/trace.sqlite"
        [ ! -f "$TRACE" ] && TRACE="$BASE_05B/tp${TP}/trace/trace.sqlite"

        if [ $B -eq 1 ]; then
            LM_HEAD="$LM_HEAD_B1"
        else
            LM_HEAD="$LM_HEAD_BN"
        fi

        echo "  batch=$B (lm_head_kernel=$LM_HEAD)"
        COMP_ARGS=""
        [ -f "$TRACE" ] && COMP_ARGS="$COMP_ARGS --asys-sqlite $TRACE"
        B1_TRACE="$BASE_05B/tp${TP}/trace/trace.sqlite"
        if [ $B -ge 2 ] && [ -f "$B1_TRACE" ]; then
            COMP_ARGS="$COMP_ARGS --sampling-trace $B1_TRACE"
        fi

        python3 "$SCRIPT_DIR/compensate_ppu.py" \
            --model-dir "$MODEL_TP" \
            --batch-size $B \
            --lm-head-kernel "$LM_HEAD" \
            --output-len 3072 \
            $COMP_ARGS \
            $COMM_ARG \
            --actual-seq-len 102400 \
            --output-json "$DIR/compensated_batch${B}.json"
        echo ""
    done
done
