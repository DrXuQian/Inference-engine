#!/bin/bash
# Chat问答: 27B FP16, TP=1 → no communication
echo "TP=1, no communication benchmark needed"
mkdir -p ./results/02_chat_27B
echo '{"method":"none","total_per_step_ms":0,"tp_size":1}' > ./results/02_chat_27B/comm.json
