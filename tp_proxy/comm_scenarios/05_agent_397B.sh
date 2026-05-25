#!/bin/bash
# Agent: Qwen 397B-A17B, TP=2 and TP=4, input=100K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4}
BASE=./results/05_agent_397B

for TP in 2 4; do
    echo "--- TP=$TP ---"
    mkdir -p "$BASE/tp${TP}"
    bash "$SCRIPT_DIR/comm_bench.sh" "$MODEL" $TP 102400 "$BASE/tp${TP}/comm.json"
    echo ""
done
