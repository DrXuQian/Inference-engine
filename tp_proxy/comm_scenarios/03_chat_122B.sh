#!/bin/bash
# Chat问答: Qwen3.5-122B-A10B, TP=1 and TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}
BASE=./results/03_chat_122B

# TP=1: no comm
mkdir -p "$BASE/tp1"
echo '{"method":"none","total_per_step_ms":0,"tp_size":1}' > "$BASE/tp1/comm.json"
echo "TP=1: no communication"

# TP=2: bench
mkdir -p "$BASE/tp2"
bash "$SCRIPT_DIR/comm_bench.sh" "$MODEL" 2 "$BASE/tp2/comm.json"
