#!/bin/bash
# Qwen3-30B-A3B GPTQ-Int4, TP=2
# No split needed — vLLM handles TP natively for GPTQ models
set -euo pipefail
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B-GPTQ-Int4}
OUT=./results/07_qwen3_30b_a3b

echo "=== Qwen3-30B-A3B-GPTQ-Int4 ==="
echo "No split/prune needed (vLLM handles TP=2 natively for GPTQ)"
echo "Model: $MODEL"
mkdir -p "$OUT/tp2"
echo "Done: $OUT/"
