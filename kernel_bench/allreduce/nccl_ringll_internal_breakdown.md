# NCCL Ring LL Internal Breakdown

This note records the first instrumented NCCL Ring LL phase breakdown. Unlike
plain nsys, this uses a locally rebuilt NCCL 2.27.5 with `clock64()` counters
inside `src/device/prims_ll.h`.

## Scope

This is an instrumented measurement, not the normal NCCL runtime:

```text
NCCL source: /root/autodl-tmp/nccl-ringll-prof
NCCL tag: v2.27.5-1
instrumented library: /root/autodl-tmp/nccl-ringll-prof/build/lib/libnccl.so.2.27.5
patched nccl-tests: /tmp/nccl-tests
NCCL patch: patches/nccl-2.27.5-ringll-prof.patch
nccl-tests patch: patches/nccl-tests-ll-prof.patch
```

The counters are aggregate thread cycles across all participating GPUs. They are
useful for phase attribution, but they are not wall-clock kernel durations. Use
nsys for absolute kernel time.

## What Was Instrumented

The instrumented points are in NCCL's LL primitive:

```text
src/device/prims_ll.h
  LLGenericOp total
  waitSend
  readLL + readLLFinish
  storeLL
  barrier
```

The test path calls exported helper functions from the instrumented NCCL:

```text
ncclLlProfResetAll(...)
ncclLlProfDumpAll(...)
```

`nccl-tests/src/common.cu` calls reset before the timed section and dump after
the timed section completes.

## Build

Build instrumented NCCL:

```bash
cd /root/autodl-tmp/nccl-ringll-prof

make -j8 src.build \
  BUILDDIR=/root/autodl-tmp/nccl-ringll-prof/build \
  CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE='-gencode=arch=compute_120,code=sm_120'
```

Build nccl-tests against the instrumented NCCL:

```bash
cd /tmp/nccl-tests

make -j8 CUDA_HOME=/usr/local/cuda \
  NCCL_HOME=/root/autodl-tmp/nccl-ringll-prof/build
```

Confirm linkage:

```bash
LD_LIBRARY_PATH=/root/autodl-tmp/nccl-ringll-prof/build/lib:$LD_LIBRARY_PATH \
  ldd /tmp/nccl-tests/build/all_reduce_perf | grep libnccl
```

Expected:

```text
libnccl.so.2 => /root/autodl-tmp/nccl-ringll-prof/build/lib/libnccl.so.2
```

## Run

Force NCCL Ring + LL and limit to one channel:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LD_LIBRARY_PATH=/root/autodl-tmp/nccl-ringll-prof/build/lib:$LD_LIBRARY_PATH \
NCCL_ALGO=Ring NCCL_PROTO=LL \
NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 \
NCCL_LL_PROF=1 \
/tmp/nccl-tests/build/all_reduce_perf \
  -b 6144 -e 6144 -g 2 -n 5 -w 0 -c 0 -m 20
```

For 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=/root/autodl-tmp/nccl-ringll-prof/build/lib:$LD_LIBRARY_PATH \
NCCL_ALGO=Ring NCCL_PROTO=LL \
NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 \
NCCL_LL_PROF=1 \
/tmp/nccl-tests/build/all_reduce_perf \
  -b 6144 -e 6144 -g 4 -n 5 -w 0 -c 0 -m 20
```

## Results

Out-of-place results from the instrumented timed section:

| Size | GPUs | all_reduce_perf time | readLL cycles | readLL % | waitSend % | storeLL % | barrier % | other % | avg readLL polls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 KB | 2 | 22.72 us | 60,706,432 | 6.13 | 1.02 | 0.45 | 1.38 | 91.02 | 1.470 |
| 1 KB | 4 | 28.69 us | 177,729,024 | 3.98 | 1.27 | 0.11 | 1.76 | 92.89 | 1.301 |
| 6 KB | 2 | 31.12 us | 392,241,568 | 33.81 | 0.82 | 2.38 | 1.18 | 61.80 | 1.568 |
| 6 KB | 4 | 33.59 us | 1,002,271,424 | 20.35 | 1.12 | 0.59 | 1.54 | 76.41 | 1.221 |

In-place was similar:

| Size | GPUs | all_reduce_perf time | readLL cycles | readLL % | waitSend % | storeLL % | barrier % | other % | avg readLL polls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 KB | 2 | 22.55 us | 59,832,384 | 6.04 | 1.00 | 0.45 | 1.38 | 91.13 | 1.462 |
| 1 KB | 4 | 28.21 us | 181,130,848 | 4.02 | 1.26 | 0.11 | 1.75 | 92.86 | 1.300 |
| 6 KB | 2 | 30.90 us | 385,706,464 | 33.76 | 0.85 | 2.42 | 1.20 | 61.78 | 1.537 |
| 6 KB | 4 | 33.20 us | 1,021,372,928 | 20.54 | 1.11 | 0.58 | 1.52 | 76.25 | 1.228 |

## Interpretation

The result changes the wording from "we only infer NCCL RingLL overhead" to:

```text
we now have an instrumented NCCL RingLL phase counter,
but it is aggregate thread-cycle attribution, not wall time.
```

What it shows:

```text
waitSend is not the bottleneck in these runs:
  avg waitSend polls per call = 0

storeLL is small:
  <= 2.42% of LLGenericOp aggregate cycles

readLL is the dominant named LL primitive at 6 KB:
  2 GPU: ~34%
  4 GPU: ~20%, but total readLL cycles are ~2.6x larger than 2 GPU

other is large:
  this includes NCCL generic loop/control/reduce work, uninstrumented parts,
  and instrumentation overhead from clock64 + atomics.
```

So the more precise statement is:

```text
NCCL RingLL = RingLL serialized read/flag dependency
              + NCCL generic implementation/control overhead
```

For this instrumented run, the extra "other" bucket is large, so it is too
coarse to call all of it framework overhead. It is the remaining LLGenericOp
work after `waitSend`, `readLL`, `storeLL`, and `barrier`, plus instrumentation
cost.

## Caveats

The inserted counters use `clock64()` and atomic adds. That makes the measured
kernel slower than production NCCL. Use the percentages and call/poll counts for
diagnosis, not the `all_reduce_perf` time as a production latency number.

The current instrumentation does not split:

```text
local input load
reduce arithmetic
loop/control
NCCL work/channel setup inside device code
instrumentation overhead
```

Those are currently included in `other`.
