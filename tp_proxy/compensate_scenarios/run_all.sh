#!/bin/bash
# Run all compensate scenarios sequentially
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "$DIR"/[0-9]*.sh; do
    echo "========================================"
    echo "Running: $(basename $script)"
    echo "========================================"
    bash "$script"
    echo ""
done
