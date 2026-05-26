#!/bin/bash
# Run vllm bench latency on a TP-split rank.
#
# vllm bench internally uses np.random.randint(10000) for dummy token IDs,
# which is safely below the split vocab_size (124160).
#
# Usage:
#   bash bench_vllm.sh /path/to/rank_0
#   bash bench_vllm.sh /path/to/rank_0 --batch-size 8 --input-len 256 --output-len 128

set -euo pipefail

RANK_DIR="${1:?Usage: $0 <rank_dir> [extra vllm bench args...]}"
shift

export TRITON_BACKENDS_IN_TREE=1

exec vllm bench latency \
    --model "${RANK_DIR}" \
    --batch-size 4 \
    --input-len 128 \
    --output-len 64 \
    --num-iters 5 \
    --num-iters-warmup 2 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    "$@"
