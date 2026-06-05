#!/bin/bash
# 代码补全: Qwen3.5-35B-A3B, TP=1 → no communication
echo "TP=1, no communication benchmark needed"
mkdir -p ./results/01_code_completion_35B
echo '{"method":"none","total_per_step_ms":0,"tp_size":1}' > ./results/01_code_completion_35B/comm.json
