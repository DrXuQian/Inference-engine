#!/bin/bash
# Benchmark all-reduce and all-gather using pccl_tools / nccl-tests.
# Reads hidden_size, vocab_size, num_hidden_layers from model config.
# Saves results to comm.json.
#
# Usage:
#   bash comm_bench.sh /path/to/model 2 ./results/comm.json
#   bash comm_bench.sh /path/to/model 4 ./results/comm.json

set -euo pipefail

MODEL_DIR="${1:?Usage: $0 <model_dir> <tp_size> <output_json>}"
TP_SIZE="${2:?}"
OUTPUT_JSON="${3:?}"

# Tool paths (override via env)
AR=${PCCL_AR:-/usr/local/PPU_SDK/pccl_tools/all_reduce_perf}
AG=${PCCL_AG:-/usr/local/PPU_SDK/pccl_tools/all_gather_perf}

# Read config
HIDDEN=$(python3 -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('text_config', cfg)
print(tc['hidden_size'])
")
VOCAB=$(python3 -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('text_config', cfg)
print(tc['vocab_size'])
")
NUM_LAYERS=$(python3 -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('text_config', cfg)
print(tc['num_hidden_layers'])
")

echo "Model: hidden=$HIDDEN, vocab=$VOCAB, layers=$NUM_LAYERS, TP=$TP_SIZE"

# AR size: decode = hidden × 2 bytes (bf16)
AR_SIZE=$((HIDDEN * 2))
# AG size: lm_head = (vocab / tp) × 2 bytes (bf16)
AG_SIZE=$(python3 -c "print(($VOCAB // $TP_SIZE) * 2)")

echo ""
echo "=== All-Reduce (decode, ${AR_SIZE} bytes) ==="
AR_OUTPUT=$($AR -b $AR_SIZE -e $AR_SIZE -f 2 -d bf16 -o sum \
    -n 500 -w 100 -g $TP_SIZE -c 0 -a 1 2>&1)
echo "$AR_OUTPUT" | grep -v "^#" | grep -v "^$" | head -3
# Parse time (column 5, us)
AR_US=$(echo "$AR_OUTPUT" | grep -v "^#" | grep -v "^$" | head -1 | awk '{print $6}')

echo ""
echo "=== All-Gather (lm_head, ${AG_SIZE} bytes) ==="
AG_OUTPUT=$($AG -b $AG_SIZE -e $AG_SIZE -f 2 -d bf16 \
    -n 300 -w 50 -g $TP_SIZE -c 0 -a 1 2>&1)
echo "$AG_OUTPUT" | grep -v "^#" | grep -v "^$" | head -3
AG_US=$(echo "$AG_OUTPUT" | grep -v "^#" | grep -v "^$" | head -1 | awk '{print $6}')

# Compute per decode step
N_AR=$((NUM_LAYERS * 2))
N_AG=2

python3 -c "
import json
ar_us = float('${AR_US}') if '${AR_US}' else 0
ag_us = float('${AG_US}') if '${AG_US}' else 0
n_ar = $N_AR
n_ag = $N_AG
total = (n_ar * ar_us + n_ag * ag_us) / 1000

print()
print(f'AR: {ar_us:.1f} us/call × {n_ar} = {n_ar*ar_us/1000:.3f} ms')
print(f'AG: {ag_us:.1f} us/call × {n_ag} = {n_ag*ag_us/1000:.3f} ms')
print(f'Total/step: {total:.3f} ms')

result = {
    'method': 'pccl_standalone',
    'ar_us': round(ar_us, 1),
    'ag_us': round(ag_us, 1),
    'ar_size_bytes': $AR_SIZE,
    'ag_size_bytes': $AG_SIZE,
    'n_ar_per_step': n_ar,
    'n_ag_per_step': n_ag,
    'total_per_step_ms': round(total, 3),
    'tp_size': $TP_SIZE,
    'hidden_size': $HIDDEN,
    'vocab_size': $VOCAB,
    'num_layers': $NUM_LAYERS,
}
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved to ${OUTPUT_JSON}')
"
