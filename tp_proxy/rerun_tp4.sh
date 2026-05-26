#!/bin/bash
# Rerun all TP=4 scenarios (397B) end-to-end
# Previous TP=4 results are invalid because split_tp2.py was hardcoded to TP=2
#
# Usage: bash tp_proxy/rerun_tp4.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
TP=4

echo "============================================"
echo "  Rerun TP=4 for Qwen3.5-397B-A17B"
echo "  (fix: split_tp2.py now supports --tp-size)"
echo "============================================"

# =============================================
# 1. Split + Bench: 05_agent_397B TP=4
# =============================================
echo ""
echo "=== [1/3] Bench: 05_agent_397B TP=4 (100K input, 30K output) ==="
TP_DIR=./results/05_agent_397B/tp4
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

# =============================================
# 2. Split + Bench: 05b_agent_hit_397B TP=4
# =============================================
echo ""
echo "=== [2/3] Bench: 05b_agent_hit_397B TP=4 (20K input, 30K output) ==="
TP_DIR=./results/05b_agent_hit_397B/tp4
mkdir -p "$TP_DIR"

python3 "$SCRIPT_DIR/split_and_prune.py" \
    --model-dir "$MODEL" --tp-size $TP \
    --gpu-memory-gb "$GPU_MEM" --max-seq-len 53248 \
    --output-dir "$TP_DIR/model"

PRUNED=$(ls -d "$TP_DIR/model/rank_0_"*L 2>/dev/null | head -1)
[ -z "$PRUNED" ] && PRUNED=$(ls -d "$TP_DIR/model/split/rank_0" 2>/dev/null | head -1)
[ -z "$PRUNED" ] && PRUNED="$MODEL"

python3 "$SCRIPT_DIR/auto_bench.py" \
    --model-dir "$PRUNED" --input-lens 20480 --output-len 30720 \
    --num-prompts 5 --gpu-mem 0.9 \
    --output-json "$TP_DIR/bench.json"

# =============================================
# 3. Trace + Compensate: TP=4 (comm.json already exists)
# =============================================
echo ""
echo "=== [3/3] Trace + Compensate: TP=4 ==="

for SCENARIO in 05_agent_397B 05b_agent_hit_397B; do
    DIR=./results/$SCENARIO/tp4
    TRACE_MODEL=$(ls -d "$DIR/model/rank_0_"*L 2>/dev/null | head -1)
    [ -z "$TRACE_MODEL" ] && continue

    if [ "$SCENARIO" = "05_agent_397B" ]; then
        INPUT_LEN=102400; OUTPUT_LEN=30720; NUM_PROMPTS=5
    else
        INPUT_LEN=20480; OUTPUT_LEN=30720; NUM_PROMPTS=5
    fi

    echo ""
    echo "--- Trace: $SCENARIO TP=$TP ---"
    bash "$SCRIPT_DIR/capture_trace.sh" "$TRACE_MODEL" $INPUT_LEN $OUTPUT_LEN "$DIR/trace" $NUM_PROMPTS

    echo ""
    echo "--- Compensate: $SCENARIO TP=$TP ---"
    python3 "$SCRIPT_DIR/compensate_ppu.py" \
        --bench-results "$DIR/bench.json" \
        --model-dir "$TRACE_MODEL" \
        --asys-sqlite "$DIR/trace/trace.sqlite" \
        --comm-json "$DIR/comm.json" \
        --output-json "$DIR/compensated.json"
done

echo ""
echo "============================================"
echo "  Done! TP=4 results:"
echo "  results/05_agent_397B/tp4/compensated.json"
echo "  results/05b_agent_hit_397B/tp4/compensated.json"
echo "============================================"
