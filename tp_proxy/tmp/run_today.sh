#!/bin/bash
# Today's tasks:
# 1. Rerun 01 code completion (1.5K input, was 15K)
# 2. Rerun 04/04b agent 122B (3K output, was 30K)
# 3. Rerun 05/05b agent 397B (3K output, was 30K) — TP=4 with fixed split
# 4. Find critical batch size (TPS drop 50%) for agent scenarios
#
# Usage: bash tp_proxy/run_today.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"

echo "============================================"
echo "  Today: updated scenarios + batch sweep"
echo "============================================"

# =============================================
# 1. Code Completion 35B (1.5K input, 50 output)
# =============================================
echo ""
echo "########################################"
echo "# 1. Code Completion 35B (1.5K→50)"
echo "########################################"
bash tp_proxy/bench_scenarios/01_code_completion_35B.sh
bash tp_proxy/trace_scenarios/01_code_completion_35B.sh
bash tp_proxy/compensate_scenarios/01_code_completion_35B.sh

# =============================================
# 2. Agent 122B (100K→3K, 20K→3K)
# =============================================
echo ""
echo "########################################"
echo "# 2. Agent 122B (output 30K→3K)"
echo "########################################"
bash tp_proxy/bench_scenarios/04_agent_122B.sh
bash tp_proxy/bench_scenarios/04b_agent_hit_122B.sh
bash tp_proxy/trace_scenarios/04_agent_122B.sh
bash tp_proxy/trace_scenarios/04b_agent_hit_122B.sh
bash tp_proxy/compensate_scenarios/04_agent_122B.sh
bash tp_proxy/compensate_scenarios/04b_agent_hit_122B.sh

# =============================================
# 3. Agent 397B (output 30K→3K, TP=4 fixed)
# =============================================
echo ""
echo "########################################"
echo "# 3. Agent 397B (output 30K→3K)"
echo "########################################"
bash tp_proxy/bench_scenarios/05_agent_397B.sh
bash tp_proxy/bench_scenarios/05b_agent_hit_397B.sh
bash tp_proxy/trace_scenarios/05_agent_397B.sh
bash tp_proxy/trace_scenarios/05b_agent_hit_397B.sh
bash tp_proxy/compensate_scenarios/05_agent_397B.sh
bash tp_proxy/compensate_scenarios/05b_agent_hit_397B.sh

# =============================================
# 4. Critical batch size search (agent scenarios)
# =============================================
echo ""
echo "########################################"
echo "# 4. Critical batch size (TPS drop 50%)"
echo "########################################"

# Use short output_len (128) for faster sweep
BATCH_SIZES="1,2,4,8,16,32,64"

for SCENARIO in 04_agent_122B 05_agent_397B; do
    for TP_DIR in ./results/$SCENARIO/tp*/; do
        [ ! -d "$TP_DIR" ] && continue
        MODEL=$(ls -d "$TP_DIR/model/rank_0_"*L 2>/dev/null | head -1)
        [ -z "$MODEL" ] && continue
        TP_NAME=$(basename "$TP_DIR")

        echo ""
        echo "--- Critical batch: $SCENARIO $TP_NAME ---"
        python3 "$SCRIPT_DIR/find_critical_batch.py" \
            --model "$MODEL" \
            --input-len 102400 --output-len 128 \
            --batch-sizes "$BATCH_SIZES" \
            --num-prompts 8 \
            --start-server --gpu-mem 0.9 \
            --output-json "$TP_DIR/critical_batch.json" \
            || echo "FAILED: $SCENARIO $TP_NAME"
    done
done

echo ""
echo "============================================"
echo "  All done! Run report:"
echo "  python tp_proxy/scripts/generate_report.py --results-dir ./results --peak-flops <FP16_TFLOPS> --peak-bw 680"
echo "============================================"
