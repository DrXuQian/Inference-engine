#!/bin/bash
# Run comm scenarios sequentially
# Usage: bash run_all.sh [LABEL]
#   LABEL=ai_station  -> run 01-06 only
#   LABEL=vla         -> run 07 only
#   (no label)        -> run all 01-07
DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="${1:-}"
FAIL=0

case "$LABEL" in
    ai_station) PATTERN="0[1-6]" ;;
    vla)        PATTERN="07" ;;
    *)          PATTERN="[0-9]" ;;
esac

for script in "$DIR"/${PATTERN}*.sh; do
    [ -f "$script" ] || continue
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
