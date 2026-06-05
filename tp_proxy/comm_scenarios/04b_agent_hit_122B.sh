#!/bin/bash
# Agent hit: Qwen3.5-122B-A10B, TP=1 and TP=2, input=20K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}
BASE=./results/04b_agent_hit_122B

mkdir -p "$BASE/tp1"
echo '{"method":"none","total_per_step_ms":0,"tp_size":1}' > "$BASE/tp1/comm.json"

mkdir -p "$BASE/tp2"
bash "$SCRIPT_DIR/comm_bench.sh" "$MODEL" 2 20480 "$BASE/tp2/comm.json"
