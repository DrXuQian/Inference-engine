#!/bin/bash
# Rerun RAG trace + compensate only (bench.json already exists)
# Usage: bash tp_proxy/rerun_rag_trace.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"
INPUT_LEN=${INPUT_LEN:-819200}
OUTPUT_LEN=${OUTPUT_LEN:-3072}
OUT=./results/06_rag_35B

PRUNED=$(ls -d "$OUT/model/rank_0_"*L 2>/dev/null | head -1)
if [ -z "$PRUNED" ]; then
    echo "ERROR: model not found in $OUT/model/rank_0_*L"
    echo "Run rerun_rag.sh first to split+bench"
    exit 1
fi

echo "============================================"
echo "  RAG trace + compensate"
echo "  Model: $PRUNED"
echo "  Input=${INPUT_LEN}, Output=${OUTPUT_LEN}"
echo "============================================"

# 1. Trace
echo ""
echo "=== [1/2] Trace ==="
bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" $INPUT_LEN $OUTPUT_LEN "$OUT/trace" 3

# 2. Compensate
echo ""
echo "=== [2/2] Compensate ==="
python3 "$SCRIPT_DIR/compensate_ppu.py" \
    --bench-results "$OUT/bench.json" \
    --model-dir "$PRUNED" \
    --asys-sqlite "$OUT/trace/trace.sqlite" \
    --output-json "$OUT/compensated.json"

echo ""
echo "============================================"
echo "  Done!"
echo "  $OUT/trace/trace.sqlite"
echo "  $OUT/compensated.json"
echo "============================================"
