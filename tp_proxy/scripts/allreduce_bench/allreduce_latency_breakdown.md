# AllReduce Latency Breakdown

This document consolidates the small-message all-reduce latency observations in
this directory. The main comparison point is 6 KB on the current 4-GPU PCIe
machine.

## Measurement Rules

Run GPU communication benchmarks serially. Do not launch multiple benchmark
processes at the same time on the same GPUs.

For CUDA event timing, capture many repeated kernel nodes into one CUDA Graph
and divide the graph time by the repeat count:

```bash
./bench_vllm_allreduce 4 20 50 6144 100 0 push 0
```

For absolute communication-kernel time, use Nsight Systems kernel duration with
CUDA Graph node tracing:

```bash
nsys profile --force-overwrite=true \
  --sample=none --cpuctxsw=none --trace=cuda \
  --cuda-graph-trace=node --cuda-trace-scope=system-wide \
  -o /tmp/report \
  ./bench_vllm_allreduce 4 0 10 6144 100 0 push 0
```

clock64() breakdowns are useful for phase attribution, but instrumentation
changes the kernel and should not replace nsys absolute time.

## Topology

Current host topology:

```text
GPU0-GPU1: NODE
GPU2-GPU3: NODE
cross pairs: SYS
EnableResizableBar: 0
RegistryDwords: ""
GrdmaPciTopoCheckOverride: 0
```

So 2-GPU measurements on GPU0/GPU1 are same-NUMA. 4-GPU measurements include
cross-NUMA `SYS` traffic.

## Current Standalone Entrypoints

| Tool | Source | Semantics | Use |
|---|---|---|---|
| `bench_vllm_allreduce` | vLLM 0.19.1 `csrc/custom_all_reduce.cuh` plus local push mode | out-of-place custom all-reduce | vLLM/push comparison |
| `standalone_ringll` | NCCL Ring LL logic extracted into one kernel | in-kernel ring LL | ring protocol phase attribution |
| `bench_breakdown` | local primitive microbench | individual barrier/read/write tests | qualitative primitive attribution |

The old local `bench_oneshot` and `bench_push_breakdown` were removed. They were
hand-written experiments and should not be used as the current source of truth.

## 6KB Current Results

Nsight Systems kernel medians, using one CUDA Graph with `repeats=100`:

| Kernel | 2 GPU | 4 GPU | 4/2 ratio | Notes |
|---|---:|---:|---:|---|
| vLLM oneshot | 4.000 us | 16.479 us | 4.1x | out-of-place pull, all ranks peer-read inputs |
| vLLM twoshot | 5.984 us | 12.704 us | 2.1x | reduce-scatter + allgather |
| Safe push | 4.448 us | 11.104 us | 2.5x | posted writes into peer scratch, local polling |
| Standalone Ring LL | 4.608 us | 10.016 us | 2.2x | separate standalone nsys median, no NCCL framework |

CUDA Graph event medians from `bench_vllm_allreduce`, also `repeats=100`:

| Kernel | 2 GPU | 4 GPU |
|---|---:|---:|
| vLLM oneshot | 4.3 us | 17.2 us |
| vLLM twoshot | 6.3 us | 13.0 us |
| Safe push | 4.8 us | 11.9 us |

Use the nsys table for conclusions about absolute GPU kernel duration. The event
table is useful as a quick harness-level check and now aligns with the large
CUDA Graph nsys runs.

## vLLM Oneshot Breakdown

vLLM oneshot is:

```text
barrier_at_start
packed peer reads from every rank input buffer
local reduce
write separate output buffer
barrier_at_end(final_sync=true)
```

The critical safety property is out-of-place output:

```text
input buffers are read-only during the all-reduce
each rank writes only its own output buffer
```

This removes the input overwrite race that existed in the old local in-place
pull benchmark.

Latency shape:

| Component | 2 GPU impact | 4 GPU impact |
|---|---|---|
| start/end barriers | fixed synchronization floor | grows with more ranks and cross-NUMA skew |
| peer reads | one peer input | three peer inputs, including `SYS` paths |
| local reduce/write | small for 6 KB | still small |
| framework overhead | removed in standalone | removed in standalone |

The 4-GPU penalty is not arithmetic. It is synchronization plus remote-read
fan-in over cross-NUMA PCIe paths.

## vLLM Twoshot Breakdown

vLLM twoshot is:

```text
barrier_at_start
reduce-scatter into per-rank temporary buffers
barrier_at_end(release/acquire)
allgather from temporary buffers
```

Compared with oneshot:

| Property | vLLM oneshot | vLLM twoshot |
|---|---|---|
| Stages | 1 read/reduce/write stage | reduce-scatter + allgather |
| Barriers | start + final | start + release/acquire middle |
| Remote access pattern | all ranks read all peer inputs | partitioned temporary-buffer exchange |
| 2-GPU 6 KB result | faster | slower |
| 4-GPU 6 KB result | slower on this topology | faster than oneshot |

At 6 KB in the large-graph measurement:

```text
2 GPU: twoshot 5.984 us vs oneshot 4.000 us
4 GPU: twoshot 12.704 us vs oneshot 16.479 us
```

So the 2-GPU choice remains oneshot. On this 4-GPU cross-NUMA PCIe topology,
twoshot avoids some of the all-rank peer-read fan-in cost and beats oneshot.

## Safe Push Breakdown

The current standalone `push` mode is not from vLLM upstream. It is a local
experimental path added to answer whether a push design can be made workable,
safe, and correct for this benchmark:

```text
barrier_at_start
each rank writes its input chunk to every rank's scratch slot
each rank polls local scratch slots
local reduce
write separate output buffer
clear local scratch slots
toggle a per-block double-buffer epoch
```

Safety properties:

| Property | Current push behavior |
|---|---|
| Input overwrite | avoided; input and output are separate |
| Scratch reuse | two scratch epochs per GPU |
| Graph repeats | every kernel node begins with a cross-rank start barrier |
| Completion check | `verify=1` validates output after repeated CUDA Graph launches |
| Zero sentinel | positive zero is reserved as empty; input positive zero is converted to negative zero before scratch write |

This is numerically correct for normal all-reduce values, but it is not a formal
bitwise-compatible all-reduce semantic because `+0` may become `-0`.

Correctness checks passed with a single CUDA Graph containing 100 repeated push
nodes:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./bench_vllm_allreduce 2 0 3 6144 100 0 push 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bench_vllm_allreduce 4 0 3 6144 100 0 push 1
```

64 KB push graph verification also passed with 20 repeated nodes:

| Size | 2 GPU event median | 4 GPU event median | Interpretation |
|---|---:|---:|---|
| 64 KB | 9.5 us | 145.4 us | correct, but 4-GPU push traffic does not scale to larger messages on this topology |

Why push helps at 4 GPU:

```text
pull oneshot: every rank performs remote reads from every peer, then final sync
push: every rank posts writes to peer scratch, then polls local memory
```

On PCIe, posted writes are cheaper than remote reads. At 4 GPUs, avoiding the
all-rank peer-read fan-in helps enough that safe push is faster than vLLM
oneshot and twoshot for 6 KB on this machine. At 2 GPUs, the start barrier and
polling overhead roughly cancel the posted-write advantage, so safe push is only
close to oneshot rather than clearly faster.

The historical bare push experiment measured about:

| Kernel | 2 GPU | 4 GPU | Notes |
|---|---:|---:|---|
| old bare push | 3.136 us | 12.448 us | removed; no graph-repeat safety |
| current safe push | 4.448 us | 11.104 us | current source of truth |

The old 2-GPU number was lower because it omitted the safety needed for repeated
graph execution.

## Ring LL Breakdown

Standalone Ring LL uses the NCCL LL protocol shape:

```text
ring neighbor transfer
readLL waits for flag from previous rank
storeLL writes data+flag to next rank
waitSend enforces credit/window progress
```

clock64 phase attribution for 6 KB:

| Phase | 2 GPU | 4 GPU | Meaning |
|---|---:|---:|---|
| total | 6.3 us | 9.7 us | rank 0 instrumented total |
| readLL spin | 4.3 us | 6.5 us | waiting for previous-rank flag/data |
| waitSend | 1.0 us | 1.2 us | credit/window wait |
| storeLL | 0.2 us | 0.2 us | posted data+flag stores |
| local load | 0.1 us | 0.1 us | local input load |
| reduce | ~0.0 us | ~0.0 us | arithmetic is negligible |
| other | 0.8 us | 1.6 us | loop/control overhead |
| readLL spin iters | 23 | 35 | more ranks increase serialized waits |

Ring LL is dominated by `readLL spin`, not by reduce arithmetic or store cost.
The 4-GPU penalty comes from serial ring progress: data and flags must advance
rank by rank.

## Primitive Microbenchmarks

`bench_breakdown` direct-event medians at 6 KB:

| Primitive | 2 GPU | 4 GPU | Interpretation |
|---|---:|---:|---|
| barrier only | 10.6 us | 8.8 us | direct-event sync benchmark; use qualitatively |
| PCIe read | 7.3 us | 7.4 us | remote read path is slower than write |
| PCIe write | 6.0 us | 5.6 us | posted write is cheaper |
| write + LL flag | 6.4 us | 6.3 us | LL flag adds little over write |
| oneshot full | 12.2 us | 25.0 us | barrier + peer-read fan-in |
| two-phase write | 9.8 us | 9.6 us | phase overhead but write path remains cheap |

These numbers are not absolute kernel truth; they are useful for explaining the
direction:

```text
remote read is expensive
posted write is cheaper
barrier/fan-in dominates small pull all-reduce
LL flag store is not the main Ring LL cost
```

## How To Understand 6KB

6 KB is 384 packed 16-byte vectors. It is still a latency-dominated message, but
it is large enough that the access pattern matters:

```text
2 GPU: synchronization dominates, so oneshot and safe push are close
4 GPU: cross-NUMA remote-read fan-in hurts oneshot heavily
4 GPU: posted writes plus local polling make safe push competitive
Ring LL: serialized readLL waits dominate even though each hop is small
```

This is why 6 KB can look different from 1 KB: the fixed barrier is still
visible, but the peer access pattern has started to matter.

## NCCL vs Standalone Ring LL

Standalone Ring LL is a protocol extraction. It does not include full NCCL
runtime overhead. Historical 1 KB measurements showed:

| Kernel | 2 GPU | 4 GPU | Notes |
|---|---:|---:|---|
| Standalone Ring LL | 4.4 us | 8.3 us | extracted kernel/protocol |
| NCCL Ring LL | ~6.5 us | 17.2 us | full NCCL path |

The extra NCCL cost comes from runtime machinery around the primitive:

```text
channel and work management
proxy/progress machinery
kernel argument and protocol setup
more conservative scheduling/generalization
```

So "Ring LL is slow" has two layers:

```text
protocol layer: serialized readLL waits
framework layer: NCCL machinery around the primitive
```

## Practical Conclusions

For this machine:

| Case | Best current interpretation |
|---|---|
| 2 GPU, 6 KB | vLLM oneshot is safest and fastest; safe push is close |
| 4 GPU, 6 KB | safe push is fastest among the current vLLM/push standalone paths |
| 4 GPU, larger messages | be cautious with all-to-all push-like traffic on this topology |
| Ring LL | predictable but serialized; dominated by `readLL spin` |
| NCCL Ring LL | includes both ring serialization and NCCL framework overhead |

The key distinction:

```text
vLLM oneshot: parallel peer-read fan-in + global barriers, out-of-place safe
vLLM twoshot: two-stage partitioned exchange, better than oneshot at 4-GPU 6 KB
Safe push: posted writes + local polling, graph-repeat safe in this benchmark
Ring LL: serialized neighbor handoff, readLL spin dominates
NCCL: robust general implementation, but adds framework overhead beyond the
      raw Ring LL primitive
```
