#!/bin/bash
# RAG仓库理解: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/06_rag_35B
MODEL=$(ls -d "$BASE/model/rank_0_"*L 2>/dev/null | head -1)

python3 "$SCRIPT_DIR/compensate_ppu.py" \
    --bench-results "$BASE/bench.json" \
    --model-dir "$MODEL" \
    --asys-sqlite "$BASE/trace/trace.sqlite" \
    --output-json "$BASE/compensated.json"
