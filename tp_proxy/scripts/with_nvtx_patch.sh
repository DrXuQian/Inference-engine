#!/bin/bash
# Run ANY python / vLLM / profiler command with the NVTX batch-size patch
# auto-applied in EVERY interpreter — the main process AND vLLM's spawned
# engine / TP-worker processes (where GPUModelRunner.execute_model actually
# runs under vLLM V1). Without this, a monkeypatch applied only in the main
# process never reaches the worker, so no 'decode bs=N' marker is emitted.
#
# How it works: it puts _sitepatch/ on PYTHONPATH. Python's site machinery
# imports _sitepatch/usercustomize.py at the startup of every interpreter
# (spawn children re-run site init), and that reads PATCH_NVTX and applies
# patch_vllm_batch_nvtx.py lazily the moment vLLM loads its model-runner module.
#
# Usage:
#   bash with_nvtx_patch.sh python3 my_vllm_script.py ...
#   bash with_nvtx_patch.sh asys profile -o trace.report -t hggc,hgtx python3 ...
#   bash with_nvtx_patch.sh nsys profile -t cuda --cuda-trace-scope=system-wide python3 ...
#
# Verify afterwards:
#   python3 inspect_nvtx.py trace.sqlite --expect <batch>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PATCH_NVTX="$SCRIPT_DIR/patch_vllm_batch_nvtx.py"
export PYTHONPATH="$SCRIPT_DIR/_sitepatch${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f "$PATCH_NVTX" ]; then
    echo "ERROR: patch not found: $PATCH_NVTX" >&2
    exit 1
fi
if [ $# -eq 0 ]; then
    sed -n '2,20p' "$0"   # print the usage header
    exit 1
fi

echo "[with_nvtx_patch] PATCH_NVTX=$PATCH_NVTX"
echo "[with_nvtx_patch] PYTHONPATH=$PYTHONPATH"
exec "$@"
