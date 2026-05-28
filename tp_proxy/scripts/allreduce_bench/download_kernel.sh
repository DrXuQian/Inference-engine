#!/bin/bash
# Download PCIe oneshot allreduce kernel from lukealonso/sglang
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
COMMIT="d39236aee635cca2725f94539358da0d1c85d8c2"
BASE="https://raw.githubusercontent.com/lukealonso/sglang/${COMMIT}/python/sglang/srt/distributed/device_communicators/pcie_allreduce"

echo "Downloading PCIe oneshot allreduce kernel..."
curl -sL "${BASE}/pcie_allreduce.cu" -o "$DIR/pcie_allreduce.cu"
curl -sL "${BASE}/__init__.py" -o "$DIR/pcie_allreduce_wrapper.py"
curl -sL "${BASE}/bench_crossover.py" -o "$DIR/bench_crossover.py"

echo "Downloaded:"
ls -la "$DIR"/*.cu "$DIR"/*.py
echo ""
echo "Source: https://github.com/lukealonso/sglang/commit/${COMMIT}"
