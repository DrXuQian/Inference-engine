#!/bin/bash
# Serve one TP-split rank with vLLM on a single GPU.
#
# Usage:
#   bash serve_rank.sh /path/to/rank_0 [port]
#
# Example:
#   bash serve_rank.sh ./output/rank_0 8000
#   bash serve_rank.sh ./output/rank_1 8001

set -euo pipefail

RANK_DIR="${1:?Usage: $0 <rank_dir> [port]}"
PORT="${2:-8000}"

if [ ! -f "${RANK_DIR}/config.json" ]; then
    echo "Error: ${RANK_DIR}/config.json not found"
    exit 1
fi

echo "Serving model from: ${RANK_DIR}"
echo "Port: ${PORT}"
echo "TP=1 (single GPU)"

exec python -m vllm.entrypoints.openai.api_server \
    --model "${RANK_DIR}" \
    --tensor-parallel-size 1 \
    --port "${PORT}" \
    --trust-remote-code \
    --dtype auto \
    --max-model-len 4096 \
    "$@"
