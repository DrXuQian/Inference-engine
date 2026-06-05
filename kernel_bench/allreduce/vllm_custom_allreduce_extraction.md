# vLLM Custom AllReduce Standalone Extraction

This directory now uses a dependency-free standalone extraction of vLLM's custom
all-reduce CUDA kernels:

```bash
nvcc -O3 -std=c++17 -arch=sm_120 bench_vllm_allreduce.cu -o bench_vllm_allreduce -lpthread
./bench_vllm_allreduce 4 20 50 6144 20 0 oneshot 1
```

For the consolidated latency breakdown across vLLM oneshot, vLLM twoshot,
the local safe push experiment, Ring LL, and NCCL, see
`allreduce_latency_breakdown.md`.

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

The standalone file also includes a local `push` mode. That mode is not from
vLLM upstream; it is an experimental posted-write path for this benchmark.

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

## Experimental Push Mode

`push` is a standalone-only path:

```text
barrier_at_start
input -> peer scratch slots by posted writes
poll local scratch slots
reduce into output
clear scratch
toggle double-buffer epoch
```

It preserves the same out-of-place input/output safety as vLLM oneshot. Scratch
reuse is protected by two epochs and a start barrier at every repeated CUDA
Graph node. `verify=1` now checks the output after the repeated graph execution,
not only before graph capture.

This is intended to be workable and correct for normal numeric inputs, not a
formal bitwise-compatible all-reduce semantic. Positive zero is reserved as the
empty scratch sentinel, so input `+0` can become `-0`.

## 6KB Result

Measurement rule: communication kernels are launched serially; nsys kernel
median is the source of truth for absolute kernel time. CUDA event and nsys runs
use one CUDA Graph containing 100 repeated kernel nodes.

CUDA Graph event timing:

| GPUs | Algo | 6KB median |
|---:|---|---:|
| 2 | oneshot | 4.3 us |
| 2 | twoshot | 6.3 us |
| 2 | push | 4.8 us |
| 4 | oneshot | 17.2 us |
| 4 | twoshot | 13.0 us |
| 4 | push | 11.9 us |

Nsight Systems kernel median with CUDA Graph node tracing:

```bash
CUDA_VISIBLE_DEVICES=0,1 nsys profile --force-overwrite=true \
  --sample=none --cpuctxsw=none --trace=cuda \
  --cuda-graph-trace=node --cuda-trace-scope=system-wide \
  -o /tmp/vllm_ar_2gpu_6kb_oneshot \
  ./bench_vllm_allreduce 2 0 10 6144 100 0 oneshot 0
```

| GPUs | Algo | Kernel count | Kernel median | Notes |
|---:|---|---:|---:|---|
| 2 | oneshot | 2000 | 4.000 us | GPU0/GPU1 same-NUMA |
| 2 | twoshot | 2000 | 5.984 us | GPU0/GPU1 same-NUMA |
| 2 | push | 2000 | 4.448 us | GPU0/GPU1 same-NUMA |
| 4 | oneshot | 4000 | 16.479 us | current machine crosses NUMA |
| 4 | twoshot | 4000 | 12.704 us | current machine crosses NUMA |
| 4 | push | 4000 | 11.104 us | current machine crosses NUMA |

Correctness checks passed:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./bench_vllm_allreduce 2 0 3 6144 100 0 oneshot 1
CUDA_VISIBLE_DEVICES=0,1 ./bench_vllm_allreduce 2 0 3 6144 100 0 twoshot 1
CUDA_VISIBLE_DEVICES=0,1 ./bench_vllm_allreduce 2 0 3 6144 100 0 push 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_vllm_allreduce 4 0 3 6144 100 0 oneshot 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_vllm_allreduce 4 0 3 6144 100 0 twoshot 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_vllm_allreduce 4 0 3 6144 100 0 push 1
```

The same push path also passed 64 KB graph verification with 20 repeated nodes,
but 4-GPU latency rose to about 145 us, so the current push experiment should be
treated as a small-message path on this topology.

## Current Interpretation

For 2 GPUs, vLLM out-of-place oneshot is safe and remains in the small-message
latency regime. For 4 GPUs on the current cross-NUMA PCIe topology, vLLM
oneshot runs correctly but pays a large peer-read fan-in cost. The local safe
push path is fastest at 6 KB because it replaces remote reads with posted writes
and local polling, but it remains an experiment outside upstream vLLM semantics.
