#!/bin/bash
# Start vllm serve for 04c/05c batch sweep (pruned single-GPU model)
#
# Usage:
#   bash 04c_serve.sh /path/to/pruned_model [port]
#   bash 04c_serve.sh ./results/04_agent_122B/tp1/model/rank_0_8L 8200
set -euo pipefail

MODEL="${1:?Usage: $0 <pruned_model_dir> [port]}"
PORT="${2:-8200}"

echo "Starting vllm serve for batch sweep..."
echo "  Model: $MODEL"
echo "  Port: $PORT"

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve "$MODEL" \
    --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9
