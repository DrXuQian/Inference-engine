#!/bin/bash
# Qwen3-30B-A3B BF16, TP=1/2/4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B}
BASE=./results/07_qwen3_30b_a3b
TPS="${TPS:-1 2 4}"

for TP in $TPS; do
    echo ""
    echo "--- TP=$TP ---"
    mkdir -p "$BASE/tp${TP}"

    if [ "$TP" = "1" ]; then
        echo '{"method":"none","total_per_step_ms":0,"decode_total_per_step_ms":0,"prefill_total_per_step_ms":0,"tp_size":1}' > "$BASE/tp1/comm.json"
        echo "TP=1: no communication"
    else
        # Measure AR/AG latency with trace-aware standalone tools.
        bash "$SCRIPT_DIR/comm_bench.sh" "$MODEL" "$TP" 1536 "$BASE/tp${TP}/comm.json"
    fi
done
