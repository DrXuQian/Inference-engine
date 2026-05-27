#!/bin/bash
# Start vllm serve and block until killed.
# Other scripts connect to this server via BASE_URL.
#
# Usage:
#   bash serve.sh /path/to/model 2              # TP=2, GPU 0,1
#   bash serve.sh /path/to/model 4 8001         # TP=4, custom port
#   GPU_IDS=2,3 bash serve.sh /path/to/model 2  # custom GPUs

set -euo pipefail

MODEL="${1:?Usage: $0 <model> <tp_size> [port]}"
TP="${2:?}"
PORT="${3:-8000}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((TP-1)))}"

echo "============================================"
echo "  vllm serve"
echo "  Model: $MODEL"
echo "  TP=$TP, GPUs=$GPU_IDS, Port=$PORT"
echo "============================================"

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve "$MODEL" \
    --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9
