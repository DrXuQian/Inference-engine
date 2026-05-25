#!/bin/bash
# Benchmark online serving (TTFT/TPOT/ITL) for a TP-split rank.
#
# Requires patching vllm/benchmarks/datasets.py to respect VLLM_BENCH_VOCAB_CAP:
#   vocab_size = int(os.environ.get('VLLM_BENCH_VOCAB_CAP', tokenizer.vocab_size))
#
# Usage:
#   bash bench_serve.sh /path/to/rank_0
#   bash bench_serve.sh /path/to/rank_0 --num-prompts 100 --random-input-len 256

set -euo pipefail

RANK_DIR="${1:?Usage: $0 <rank_dir> [extra bench args...]}"
shift

export TRITON_BACKENDS_IN_TREE=1

# Read split vocab_size from config and export as cap
SPLIT_VOCAB=$(python3 -c "
import json
with open('${RANK_DIR}/config.json') as f:
    c = json.load(f)
print(c.get('text_config',c)['vocab_size'])
")
export VLLM_BENCH_VOCAB_CAP=$SPLIT_VOCAB
echo "VLLM_BENCH_VOCAB_CAP=$SPLIT_VOCAB"

# Start server
python3 -m vllm.entrypoints.openai.api_server \
    --model "${RANK_DIR}" \
    --tensor-parallel-size 1 --dtype auto --max-model-len 512 \
    --trust-remote-code --enforce-eager --gpu-memory-utilization 0.9 \
    --port 8000 > /tmp/vllm_server.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

echo "Waiting for server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready (${i}s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died. Check /tmp/vllm_server.log"; exit 1
    fi
    sleep 2
done

# Run bench
exec vllm bench serve \
    --model "${RANK_DIR}" \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 64 \
    --num-prompts 16 \
    --backend openai \
    --endpoint /v1/completions \
    --base-url http://localhost:8000 \
    --ignore-eos \
    --trust-remote-code \
    "$@"
