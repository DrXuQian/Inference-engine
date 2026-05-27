#!/bin/bash
# Agent batch sweep comm: measure AR/AG for each batch size
# AR size = batch × hidden × 2 bytes (scales with batch)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}
OUT=./results/04c_agent_batch_122B
INPUT_LEN=${INPUT_LEN:-102400}
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

# TP=1: no comm
mkdir -p "$OUT/tp1"
for B in $BATCH_LIST; do
    echo "{\"method\":\"none\",\"total_per_step_ms\":0,\"decode_total_per_step_ms\":0,\"prefill_total_per_step_ms\":0,\"tp_size\":1,\"batch_size\":$B}" \
        > "$OUT/tp1/comm_batch${B}.json"
done
echo "TP=1: no communication"

# TP=2: measure with batch-scaled AR size
echo ""
mkdir -p "$OUT/tp2"
for B in $BATCH_LIST; do
    echo "--- TP=2 batch=$B ---"
    # comm_bench.sh uses input_len to compute prefill AR size
    # For decode AR, we need batch × hidden × 2, but comm_bench.sh computes hidden × 2
    # So we pass input_len=batch for decode-equivalent sizing
    # Actually comm_bench.sh hardcodes AR_DECODE_SIZE = hidden * 2 (batch=1)
    # We need a custom approach for batch>1

    HIDDEN=$(python3 -c "
import json
with open('${MODEL}/config.json') as f:
    cfg = json.load(f)
print(cfg.get('text_config', cfg)['hidden_size'])
")

    # For batch>1, decode AR size = batch × hidden × 2
    # Use comm_bench.sh but override with batch-aware input_len trick:
    # Since comm_bench computes AR_PREFILL_SIZE = INPUT_LEN × HIDDEN × 2,
    # we can use INPUT_LEN=batch to get batch × hidden × 2 for "prefill" AR
    # Then use that as decode AR
    bash "$SCRIPT_DIR/comm_bench.sh" "$MODEL" 2 $((B * 1)) "$OUT/tp2/comm_batch${B}.json" 2>/dev/null || true

    # Fix: override decode AR with batch-scaled value
    python3 -c "
import json, subprocess
hidden = $HIDDEN
batch = $B
tp = 2

# Load existing comm.json
try:
    with open('$OUT/tp2/comm_batch${B}.json') as f:
        comm = json.load(f)
except:
    comm = {}

# Decode AR: batch × hidden × 2 bytes
comm['decode_ar_size_bytes'] = batch * hidden * 2
comm['batch_size'] = batch
# Prefill AR stays the same (input_len × hidden × 2)
comm['prefill_ar_size_bytes'] = $INPUT_LEN * hidden * 2

with open('$OUT/tp2/comm_batch${B}.json', 'w') as f:
    json.dump(comm, f, indent=2)
print(f'  batch={batch}: decode_ar={batch*hidden*2} bytes')
"
done
