# AllReduce Benchmarks

This directory contains standalone communication benchmarks for comparing vLLM
custom all-reduce, a local safe push experiment, standalone Ring LL, and NCCL.

## Main Files

| File | Purpose |
|---|---|
| `bench_vllm_allreduce.cu` | vLLM oneshot/twoshot extraction plus local `push` mode |
| `allreduce_latency_breakdown.md` | unified latency data and protocol analysis |
| `ringll_vs_oneshot.md` | focused explanation of why Ring LL can lose to oneshot |
| `vllm_custom_allreduce_extraction.md` | details of the vLLM standalone extraction |
| `standalone_ringll.cu` | standalone NCCL Ring LL protocol extraction |
| `bench_breakdown.cu` | primitive barrier/read/write breakdown benchmark |

## Build Push/Oneshot/Twoshot Benchmark

```bash
cd /root/autodl-tmp/Inference-engine/tp_proxy/scripts/allreduce_bench

nvcc -O3 -std=c++17 -arch=sm_120 bench_vllm_allreduce.cu \
  -o bench_vllm_allreduce -lpthread
```

## Test Safe Push

Run benchmarks serially. Do not run multiple GPU communication benchmark
processes at the same time.

Correctness, using one CUDA Graph with 100 repeated push nodes:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./bench_vllm_allreduce 2 0 3 6144 100 0 push 1

CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_vllm_allreduce 4 0 3 6144 100 0 push 1
```

CUDA event timing, also with 100 repeated graph nodes:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./bench_vllm_allreduce 2 20 50 6144 100 0 push 0

CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_vllm_allreduce 4 20 50 6144 100 0 push 0
```

Nsight Systems absolute kernel timing:

```bash
CUDA_VISIBLE_DEVICES=0,1 nsys profile --force-overwrite=true \
  --sample=none --cpuctxsw=none --trace=cuda \
  --cuda-graph-trace=node --cuda-trace-scope=system-wide \
  -o /tmp/push_2gpu_6kb_r100 \
  ./bench_vllm_allreduce 2 0 10 6144 100 0 push 0

CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile --force-overwrite=true \
  --sample=none --cpuctxsw=none --trace=cuda \
  --cuda-graph-trace=node --cuda-trace-scope=system-wide \
  -o /tmp/push_4gpu_6kb_r100 \
  ./bench_vllm_allreduce 4 0 10 6144 100 0 push 0
```

Export and summarize an nsys report:

```bash
nsys export --type sqlite --force-overwrite=true \
  -o /tmp/push_4gpu_6kb_r100.sqlite /tmp/push_4gpu_6kb_r100.nsys-rep

sqlite3 /tmp/push_4gpu_6kb_r100.sqlite "
WITH k AS (
  SELECT (end-start)/1000.0 AS us
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  JOIN StringIds sid ON demangledName=sid.id
  WHERE sid.value LIKE '%cross_device_reduce_push_safe%'
),
r AS (
  SELECT us,row_number() OVER (ORDER BY us) rn,count(*) OVER () cnt FROM k
)
SELECT count(*), round(min(us),3), round(avg(us),3),
       round((SELECT us FROM r WHERE rn=(cnt+1)/2),3),
       round(max(us),3)
FROM k;"
```

## Current 6KB Reference

Nsight Systems kernel median, one CUDA Graph with 100 repeated nodes:

| Kernel | 2 GPU | 4 GPU |
|---|---:|---:|
| vLLM oneshot | 4.000 us | 16.479 us |
| vLLM twoshot | 5.984 us | 12.704 us |
| safe push | 4.448 us | 11.104 us |
| standalone Ring LL | 4.608 us | 10.016 us |
