#!/bin/bash
# One-time patch: add VLLM_BENCH_VOCAB_CAP support to vllm bench datasets.
# Run once per environment. Idempotent (safe to run multiple times).
#
# Usage: bash patch_vllm_vocab_cap.sh

set -euo pipefail

DATASETS_PY=$(python3 -c "import vllm.benchmarks.datasets; print(vllm.benchmarks.datasets.__file__)")
echo "Patching: $DATASETS_PY"

# Check if already patched
if grep -q "VLLM_BENCH_VOCAB_CAP" "$DATASETS_PY"; then
    echo "Already patched. Nothing to do."
    exit 0
fi

# Add 'import os' if not present
if ! grep -q "^import os" "$DATASETS_PY"; then
    sed -i '1s/^/import os\n/' "$DATASETS_PY"
    echo "Added: import os"
fi

# Replace all occurrences
COUNT=$(grep -c "vocab_size = tokenizer.vocab_size" "$DATASETS_PY" || true)
sed -i "s/vocab_size = tokenizer.vocab_size/vocab_size = int(os.environ.get('VLLM_BENCH_VOCAB_CAP', tokenizer.vocab_size))/g" "$DATASETS_PY"
echo "Patched $COUNT occurrences."
echo "Done. Set VLLM_BENCH_VOCAB_CAP=<split_vocab_size> when running vllm bench serve."
