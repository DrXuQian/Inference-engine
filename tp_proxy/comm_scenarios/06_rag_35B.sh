#!/bin/bash
# RAG: Qwen3.5-35B-A3B, TP=1 → no communication
echo "TP=1, no communication benchmark needed"
mkdir -p ./results/06_rag_35B
echo '{"method":"none","total_per_step_ms":0,"tp_size":1}' > ./results/06_rag_35B/comm.json
