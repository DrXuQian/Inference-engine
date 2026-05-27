#!/bin/bash
# Start vllm serve with TP on multiple GPUs.
#
# Usage:
#   bash start_server.sh /path/to/model 2          # TP=2, GPU 0,1
#   bash start_server.sh /path/to/model 4          # TP=4, GPU 0,1,2,3
#   bash start_server.sh /path/to/model 2 8001     # custom port
#   GPU_IDS=2,3 bash start_server.sh /path/to/model 2  # custom GPUs

MODEL="${1:?Usage: $0 <model_dir> <tp_size> [port]}"
TP="${2:?}"
PORT="${3:-8000}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((TP-1)))}"

echo "============================================"
echo "  vllm serve"
echo "  Model: $MODEL"
echo "  TP=$TP, GPUs=$GPU_IDS, Port=$PORT"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU_IDS \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve "$MODEL" \
    --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9
