#!/bin/bash
# Qwen3-30B-A3B BF16, TP=1 (no comm) and TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B}
BASE=./results/07_qwen3_30b_a3b

# TP=1: no communication
mkdir -p "$BASE/tp1"
echo '{"method":"none","total_per_step_ms":0,"decode_total_per_step_ms":0,"prefill_total_per_step_ms":0,"tp_size":1}' > "$BASE/tp1/comm.json"
echo "TP=1: no communication"

# TP=2: measure AR/AG latency with trace (kernel-level accuracy)
mkdir -p "$BASE/tp2"
bash "$SCRIPT_DIR/comm_bench.sh" "$MODEL" 2 1536 "$BASE/tp2/comm.json"
