#!/bin/bash
# Agent batch sweep benchmark against a RUNNING server.
#
# Start server first:
#   bash 04c_serve.sh $(bash ../scripts/get_model_path.sh ./results/04_agent_122B/tp1/model)
#
# Then run this:
#   bash 04c_agent_batch_122B.sh 1              # TP=1
#   bash 04c_agent_batch_122B.sh 2 8201         # TP=2, custom port
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
TP="${1:?Usage: $0 <tp_size> [port]}"
PORT="${2:-8200}"
BASE_URL="http://127.0.0.1:${PORT}"
BASE_04=./results/04_agent_122B
OUT=./results/04c_agent_batch_122B
INPUT_LEN=${INPUT_LEN:-102400}
OUTPUT_LEN=${OUTPUT_LEN:-3072}
NUM_PROMPTS=${NUM_PROMPTS:-40}
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

TP_DIR="$OUT/tp${TP}"
mkdir -p "$TP_DIR"

PRUNED=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
[ -z "$PRUNED" ] && echo "ERROR: model not found in $BASE_04/tp${TP}/model" && exit 1

# Wait for server
echo "Checking server at $BASE_URL ..."
for i in $(seq 1 300); do
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Server ready"
        break
    fi
    [ $i -eq 300 ] && echo "ERROR: server not ready" && exit 1
    sleep 2
done

echo "=== Agent Batch Sweep TP=$TP ==="
echo "Model: $PRUNED"
echo "Input=$INPUT_LEN, Output=$OUTPUT_LEN, Batch: $BATCH_LIST"

for B in $BATCH_LIST; do
    echo ""
    echo "  [TP=$TP batch=$B]"
    python3 "$SCRIPT_DIR/auto_bench.py" \
        --model-dir "$PRUNED" \
        --input-lens $INPUT_LEN \
        --output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --batch-size $B \
        --gpu-mem 0.9 \
        --output-json "$TP_DIR/bench_batch${B}.json"
done

echo ""
echo "Done: $TP_DIR/"
