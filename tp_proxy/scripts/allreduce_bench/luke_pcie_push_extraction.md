# Luke/SGLang PCIe Push Oneshot Extraction

This directory now contains an extracted push-style PCIe oneshot all-reduce path in
`bench_oneshot.cu`:

```bash
./bench_oneshot 4 50 200 65536 20 5 push 1
./bench_push_breakdown 2 1024 100 200
```

The final argument enables a one-launch correctness check before timing. Pass
`0` to disable it for profiling-only runs.

## Source Trail

The rtx6kpro notes point at Luke Alonso's SGLang commit:

```text
429670d12149d5e417effcbe960b9a6d9b029f1f Add PCIe oneshot all-reduce backend
```

That commit wires SGLang to `b12x.distributed.PCIeOneshotAllReduce`, but the
actual b12x CUDA source (`pcie_allreduce.cu` / `pcie_oneshot.cu`) is not present
in the local checkout, and the documented raw GitHub URL currently returns 404.

The extracted local kernel therefore follows the equivalent SGLang JIT push
implementation:

```text
/root/autodl-tmp/sglang/python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh
```

This is a protocol extraction for standalone benchmarking. It does not include
the unavailable b12x extension wrapper, fused RMSNorm/add epilogues, or startup
crossover autotuning.

## Protocol

`algo=push` is different from the old local `oneshot` benchmark:

1. Each rank reads its own input.
2. Each rank writes that input into every peer's scratch buffer.
3. Each rank polls its local scratch slots until all peers have written.
4. Each rank reduces only local scratch data.
5. Each rank clears the scratch slots back to positive zero.
6. A per-block two-entry epoch toggles the scratch region for the next launch.

The scratch layout is:

```text
push_buffer[dst_rank][epoch][src_rank][payload]
```

The positive-zero sentinel mirrors the SGLang push design: payload positive zeros
are converted to negative zero before publish, so polling can use positive zero
as "not written yet" without an extra flag array.

## Why This Matters

The previous local pull oneshot reads peer input buffers directly and writes the
result back in place. On cross-NUMA PCIe, skew can expose an input/output alias
race: one rank may overwrite its input before another rank has peer-read the
original value.

The push version avoids that specific race because peer GPUs consume scratch
slots, not the source rank's input buffer.

## Measurement Rule

For communication kernel timing, run kernels serially and use `nsys`/`asys`
kernel durations for absolute time. CUDA event timing can include launch and
framework overhead and should not be used as the sole source of truth.

Example profiling command:

```bash
nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
  --trace=cuda --cuda-graph-trace=node --cuda-trace-scope=system-wide \
  -o push_64kb_4gpu \
  ./bench_oneshot 4 0 20 65536 1 0 push 0
```

## 2-GPU Alignment Result

Target: align with the local vLLM-style custom `oneshot` path on two GPUs.

Environment note from this machine:

```text
GPU0-GPU1 topology: NODE
ForceP2P/RegistryDwords: not enabled
```

CUDA Graph timing, `CUDA_VISIBLE_DEVICES=0,1`, `warmup=20`, `iters=50`,
`repeats=20`, `verify=0`:

| Size | vLLM-style oneshot | Extracted push | Result |
|---:|---:|---:|---|
| 1 KB | 5.1 us | 3.3 us | push faster |
| 4 KB | 5.1 us | 3.5 us | push faster |
| 8 KB | 5.3 us | 3.7 us | push faster |
| 16 KB | 6.0 us | 4.8 us | push faster |
| 32 KB | 6.4 us | 5.0 us | push faster |
| 64 KB | 7.6 us | 5.7 us | push faster |
| 128 KB | 9.9 us | 8.0 us | push faster |
| 256 KB | 15.1 us | 12.5 us | push faster |
| 512 KB | 25.1 us | 22.3 us | push faster |
| 1 MB | 45.3 us | 41.6 us | push faster |

Nsight Systems kernel median with `--cuda-graph-trace=node`:

| Size | vLLM-style oneshot kernel | Extracted push kernel |
|---:|---:|---:|
| 1 KB | 4.704 us | 3.104 us |
| 64 KB | 7.808 us | 5.120 us |

So for the 2-GPU target, the extracted push path is aligned with, and currently
faster than, the local vLLM-style custom oneshot path.

## Push Kernel Breakdown Results

Absolute kernel time comes from `nsys` on the original `bench_oneshot push`
kernel, using CUDA Graph node tracing:

```bash
CUDA_VISIBLE_DEVICES=0,1 nsys profile --force-overwrite=true \
  --sample=none --cpuctxsw=none --trace=cuda \
  --cuda-graph-trace=node --cuda-trace-scope=system-wide \
  -o /tmp/push_detail_2gpu_1kb \
  ./bench_oneshot 2 0 50 1024 1 0 push 0
```

| GPUs | Size | nsys kernel median | Notes |
|---:|---:|---:|---|
| 2 | 1 KB | 3.104 us | same-NUMA pair GPU0/GPU1 |
| 2 | 64 KB | 5.120 us | same-NUMA pair GPU0/GPU1 |
| 4 | 1 KB | 7.104 us | current machine crosses NUMA |
| 4 | 64 KB | 155.966 us | current machine crosses NUMA |

The detailed phase attribution comes from `bench_push_breakdown.cu`, a
clock64-instrumented version of the same push hot path. These phase values are
for attribution, not the final absolute time. The instrumented kernel adds
`clock64()` and timing writes, so the original nsys numbers above remain the
absolute source of truth. Phase max values are also not additive: `sync_epoch`
mostly shows fast threads waiting at the block tail for threads that are still
stuck in `poll`.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./bench_push_breakdown 2 1024 100 200
CUDA_VISIBLE_DEVICES=0,1 ./bench_push_breakdown 2 65536 100 200
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_push_breakdown 4 1024 100 200
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_push_breakdown 4 65536 100 200
```

| GPUs | Size | total | load_clear | store_local | store_remote | poll | reduce | write_result | clear | sync_epoch | poll iters |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 KB | 4.645 us | 0.183 us | 0.014 us | 0.014 us | 4.010 us | 0.017 us | 0.013 us | 0.003 us | 4.502 us | 20.00 |
| 2 | 64 KB | 7.541 us | 0.710 us | 0.060 us | 0.052 us | 6.438 us | 0.054 us | 0.044 us | 0.006 us | 1.626 us | 22.39 |
| 4 | 1 KB | 6.518 us | 0.277 us | 0.014 us | 0.041 us | 5.778 us | 0.037 us | 0.013 us | 0.006 us | 6.358 us | 24.00 |
| 4 | 64 KB | 141.133 us | 0.694 us | 24.567 us | 20.526 us | 139.636 us | 0.084 us | 0.040 us | 0.017 us | 115.601 us | 265.68 |

The fine-grained result is clear: push latency is dominated by local scratch
polling. The posted writes themselves are cheap for 2 GPUs. For 4 GPUs at
64 KB, the all-to-all push traffic crosses `SYS` links on this machine; posted
write backpressure and scratch visibility delay explode, and `poll` becomes the
entire kernel.

## Kernel Breakdown Reanalysis

The earlier breakdown still explains the local vLLM-style pull `oneshot`, but it
does not explain the extracted `push` kernel. They are different protocols.

### Pull oneshot

The old local `oneshot` path in `bench_oneshot.cu` does:

```text
start barrier
peer-read all input buffers
write result in place
completion barrier
```

For 2 GPUs, the previous primitive breakdown maps well:

```text
barrier only ~= 3.7 us
peer read    ~= 1.8 us
kernel median ~= 4.7 us
```

So pull `oneshot` is mainly:

```text
global synchronization + one remote PCIe read + local reduce/write
```

The final completion barrier is important for buffer lifetime, but the visible
small-message latency is dominated by the start synchronization plus the remote
read return path.

### Extracted push oneshot

The extracted `push` path does:

```text
local input read
posted write to every peer's scratch slot
poll local scratch slots
local reduce
clear scratch slots
epoch toggle
```

There is no explicit global `multi_gpu_barrier` in the hot path. The dependency
is encoded by scratch-slot visibility: the receiver polls local memory until the
peer's posted write has arrived.

For 2 GPUs, the measured kernel medians are:

```text
1 KB:  push 3.104 us vs pull oneshot 4.704 us
64 KB: push 5.120 us vs pull oneshot 7.808 us
```

The important replacement is:

```text
pull: global barrier + remote read round trip
push: posted remote write + local poll
```

PCIe writes are posted, so the sender does not pay a per-cacheline read-return
latency. The receiver waits by repeatedly loading its local scratch slot. This is
why the push kernel can beat pull `oneshot` on two GPUs even though it performs
extra local scratch traffic and clears the slots after reduction.

### Ring LL Comparison

Ring LL is still a different latency shape:

```text
serial readLL waits along the ring
```

For 4 GPUs, Ring LL accumulates multiple serialized `readLL` waits, which is why
the previous breakdown found `readLL spin` dominating. The push kernel does not
forward data hop by hop; all ranks publish directly to every destination scratch
buffer. That removes ring-hop serialization but creates all-to-all posted-write
traffic.

### 4-GPU Caveat

On the current machine, all 4 GPUs are not same-NUMA:

```text
GPU0-GPU1: NODE
GPU2-GPU3: NODE
cross pair: SYS
```

So 4-GPU push performance is not a clean same-NUMA result. It includes
cross-socket posted writes and scratch visibility over `SYS` links, with
ForceP2P not enabled. The two-GPU result on `GPU0,GPU1` is the clean alignment
target for this extraction.
