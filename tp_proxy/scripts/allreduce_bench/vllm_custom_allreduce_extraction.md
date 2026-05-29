# vLLM Custom AllReduce Standalone Extraction

This directory now uses a dependency-free standalone extraction of vLLM's custom
all-reduce CUDA kernels:

```bash
nvcc -O3 -std=c++17 -arch=sm_120 bench_vllm_allreduce.cu -o bench_vllm_allreduce -lpthread
./bench_vllm_allreduce 4 20 50 6144 20 0 oneshot 1
```

## Source

The kernel logic was extracted from the installed vLLM version:

```text
vllm 0.19.1
git tag: v0.19.1
commit: b1388b1fbf5aaef47937fabe98931211684666a6
source: csrc/custom_all_reduce.cuh
```

The standalone file keeps the vLLM CUDA pieces that matter for benchmarking:

```text
Signal { start, end, _flag }
barrier_at_start
barrier_at_end
cross_device_reduce_1stage
cross_device_reduce_2stage
packed 16B load/reduce/store path
```

It removes the vLLM/PyTorch op wrapper, IPC handle exchange, Python policy, and
NCCL fallback. All GPU pointers are registered directly inside one process.

## Safety Difference

The removed local benchmark used an in-place pull shape:

```text
read peer input
write result back into the same input buffer
```

That is not safe as a general all-reduce because a fast rank can overwrite its
input before a slow peer has read the original value.

The vLLM version is out-of-place:

```text
peer reads input buffers
each rank writes only its own output buffer
```

So the input overwrite race is gone. The standalone harness allocates separate
`input` and `output` buffers on every rank to preserve that vLLM semantic.

## 4-GPU Extension

vLLM's CUDA kernels support 2/4/6/8 ranks, but the Python communicator disables
custom all-reduce for more than two PCIe-only GPUs unless the topology is fully
connected. This standalone benchmark intentionally removes that policy gate so
we can measure the 4-GPU PCIe case directly.

The `auto` policy here follows vLLM's kernel threshold choice as if 4 GPUs were
allowed:

```text
2 GPU: always oneshot
<=4 GPU and bytes < 512KB: oneshot
<=8 GPU and bytes < 256KB: oneshot
otherwise: twoshot
```

## 6KB Result

Measurement rule: communication kernels are launched serially; nsys kernel
median is the source of truth for absolute kernel time.

CUDA Graph event timing:

| GPUs | Algo | 6KB median |
|---:|---|---:|
| 2 | oneshot | 4.7 us |
| 2 | twoshot | 6.7 us |
| 4 | oneshot | 17.1 us |
| 4 | twoshot | 13.4 us |

Nsight Systems kernel median with CUDA Graph node tracing:

```bash
CUDA_VISIBLE_DEVICES=0,1 nsys profile --force-overwrite=true \
  --sample=none --cpuctxsw=none --trace=cuda \
  --cuda-graph-trace=node --cuda-trace-scope=system-wide \
  -o /tmp/vllm_ar_2gpu_6kb_oneshot \
  ./bench_vllm_allreduce 2 0 50 6144 1 0 oneshot 0
```

| GPUs | Algo | Kernel count | Kernel median | Notes |
|---:|---|---:|---:|---|
| 2 | oneshot | 100 | 4.320 us | GPU0/GPU1 same-NUMA |
| 4 | oneshot | 200 | 12.928 us | current machine crosses NUMA |

Correctness checks passed:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./bench_vllm_allreduce 2 0 3 6144 1 0 oneshot 1
CUDA_VISIBLE_DEVICES=0,1 ./bench_vllm_allreduce 2 0 3 6144 1 0 twoshot 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_vllm_allreduce 4 0 3 6144 1 0 oneshot 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_vllm_allreduce 4 0 3 6144 1 0 twoshot 1
```

## Current Interpretation

For 2 GPUs, vLLM out-of-place oneshot is safe and remains in the small-message
latency regime. For 4 GPUs on the current cross-NUMA PCIe topology, the kernel
runs correctly, but latency is much higher than the 2-GPU case. This validates
why upstream vLLM keeps the Python policy conservative for PCIe-only world sizes
larger than two.
