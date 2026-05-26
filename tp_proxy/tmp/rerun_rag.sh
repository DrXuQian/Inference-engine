#!/bin/bash
# Rerun RAG scenario (06_rag_35B) end-to-end
# Usage: bash tp_proxy/rerun_rag.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
INPUT_LEN=${INPUT_LEN:-819200}
OUTPUT_LEN=${OUTPUT_LEN:-3072}
OUT=./results/06_rag_35B

echo "============================================"
echo "  RAG: Qwen3.5-35B-A3B GPTQ-Int4"
echo "  Input=${INPUT_LEN}, Output=${OUTPUT_LEN}"
echo "============================================"

# 1. Split + Bench
echo ""
echo "=== [1/3] Split + Bench ==="
python3 "$SCRIPT_DIR/split_and_prune.py" \
    --model-dir "$MODEL" --tp-size 1 \
    --gpu-memory-gb "$GPU_MEM" --max-seq-len $((INPUT_LEN + OUTPUT_LEN + 2048)) \
    --output-dir "$OUT/model"

PRUNED=$(ls -d "$OUT/model/rank_0_"*L 2>/dev/null | head -1)
[ -z "$PRUNED" ] && PRUNED="$MODEL"

python3 "$SCRIPT_DIR/auto_bench.py" \
    --model-dir "$PRUNED" --input-lens $INPUT_LEN --output-len $OUTPUT_LEN \
    --num-prompts 3 --gpu-mem 0.9 \
    --output-json "$OUT/bench.json"

# 2. Trace
echo ""
echo "=== [2/3] Trace ==="
bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" $INPUT_LEN $OUTPUT_LEN "$OUT/trace" 3

# 3. Compensate
echo ""
echo "=== [3/3] Compensate ==="
python3 "$SCRIPT_DIR/compensate_ppu.py" \
    --bench-results "$OUT/bench.json" \
    --model-dir "$PRUNED" \
    --asys-sqlite "$OUT/trace/trace.sqlite" \
    --output-json "$OUT/compensated.json"

echo ""
echo "============================================"
echo "  Done! Results:"
echo "  $OUT/bench.json"
echo "  $OUT/trace/trace.sqlite"
echo "  $OUT/compensated.json"
echo "============================================"
