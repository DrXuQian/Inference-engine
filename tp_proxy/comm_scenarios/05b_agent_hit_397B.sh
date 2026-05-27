#!/bin/bash
# Agent hit: Qwen 397B-A17B, TP=2 and TP=4, input=20K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4}
BASE=./results/05b_agent_hit_397B

for TP in 2; do
    echo "--- TP=$TP ---"
    mkdir -p "$BASE/tp${TP}"
    bash "$SCRIPT_DIR/comm_bench.sh" "$MODEL" $TP 20480 "$BASE/tp${TP}/comm.json"
    echo ""
done
