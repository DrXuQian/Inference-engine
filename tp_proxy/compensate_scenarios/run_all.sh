#!/bin/bash
# Run all compensate scenarios sequentially
DIR="$(cd "$(dirname "$0")" && pwd)"
FAIL=0

for script in "$DIR"/[0-9]*.sh; do
    echo "========================================"
    echo "Running: $(basename $script)"
    echo "========================================"
    set +e
    bash "$script" 2>&1
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "FAILED: $(basename $script) (exit code $rc)"
        FAIL=$((FAIL + 1))
    else
        echo "SUCCESS: $(basename $script)"
    fi
    echo ""
done

if [ $FAIL -gt 0 ]; then
    echo "$FAIL scenario(s) failed."
    exit 1
fi
