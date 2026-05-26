#!/bin/bash
# Convert all trace.sqlite files to CSV
# Usage: bash export_all_sqlite.sh [results_dir]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${1:-./results}"

SQLITE_FILES=(
    "$RESULTS_DIR/01_code_completion_35B/trace/trace.sqlite"
    "$RESULTS_DIR/02_chat_27B/trace/trace.sqlite"
    "$RESULTS_DIR/03_chat_122B/tp1/trace/trace.sqlite"
    "$RESULTS_DIR/03_chat_122B/tp2/trace/trace.sqlite"
    "$RESULTS_DIR/04_agent_122B/tp1/trace/trace.sqlite"
    "$RESULTS_DIR/04_agent_122B/tp2/trace/trace.sqlite"
    "$RESULTS_DIR/04b_agent_hit_122B/tp1/trace/trace.sqlite"
    "$RESULTS_DIR/04b_agent_hit_122B/tp2/trace/trace.sqlite"
    "$RESULTS_DIR/05_agent_397B/tp2/trace/trace.sqlite"
    "$RESULTS_DIR/05_agent_397B/tp4/trace/trace.sqlite"
    "$RESULTS_DIR/05b_agent_hit_397B/tp2/trace/trace.sqlite"
    "$RESULTS_DIR/05b_agent_hit_397B/tp4/trace/trace.sqlite"
    "$RESULTS_DIR/06_rag_35B/trace/trace.sqlite"
)

TOTAL=0
OK=0
FAIL=0

for db in "${SQLITE_FILES[@]}"; do
    if [ ! -f "$db" ]; then
        echo "SKIP: $db (not found)"
        continue
    fi
    TOTAL=$((TOTAL + 1))
    CSV_DIR="$(dirname "$db")/csv"
    echo "========================================"
    echo "Exporting: $db"
    echo "      To: $CSV_DIR"
    echo "========================================"
    if python "$SCRIPT_DIR/sqlite_to_csv.py" "$db" -o "$CSV_DIR"; then
        OK=$((OK + 1))
    else
        echo "FAILED: $db"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

echo "========================================"
echo "Done: $OK/$TOTAL succeeded, $FAIL failed"
echo "========================================"
