# Ring LL vs Oneshot

This note explains why Ring LL can look much worse than custom oneshot for small
all-reduce messages, and why NCCL Ring LL is slower than the standalone Ring LL
extraction.

## Short Answer

For small messages, the difference is mostly protocol latency, not arithmetic:

```text
oneshot: one global sync, then every rank reads all peers in parallel
Ring LL: data/flags advance through the ring one neighbor hop at a time
NCCL Ring LL: Ring LL protocol plus NCCL generic/control work around the primitive
```

So custom oneshot wins when the message is small enough that parallel peer reads
finish before serialized ring handoff would.

## Current Small-Message Data

6 KB, Nsight Systems kernel median, large CUDA Graph with repeated nodes:

| Kernel | 2 GPU | 4 GPU | Notes |
|---|---:|---:|---|
| vLLM oneshot | 4.000 us | 16.479 us | out-of-place pull, all ranks peer-read inputs |
| vLLM twoshot | 5.984 us | 12.704 us | reduce-scatter + allgather |
| safe push | 4.448 us | 11.104 us | local experimental posted-write path |
| Standalone Ring LL | 4.608 us | 10.016 us | separate standalone nsys run |

Historical 1 KB measurements:

| Kernel | 2 GPU | 4 GPU | Notes |
|---|---:|---:|---|
| Oneshot | 4.7 us | 6.9 us | local custom oneshot result |
| Standalone Ring LL | 4.4 us | 8.3 us | extracted Ring LL protocol |
| NCCL Ring LL | ~6.5 us | 17.2 us | full NCCL path |

Important nuance: standalone Ring LL is not always slower than every oneshot
variant at every size. The robust conclusion is:

```text
TP=2: standalone Ring LL and oneshot are close
4 GPU tiny message: oneshot often wins because ring serialization grows
full NCCL Ring LL: slower than standalone due to generic/control overhead too
```

## TP=2 Case

With 2 GPUs, Ring LL degenerates to the simplest possible ring:

```text
rank0 <-> rank1
```

There is only one neighbor and only one remote data dependency per phase. That
means standalone Ring LL should be close to oneshot:

```text
1 KB historical: standalone Ring LL 4.4 us, oneshot 4.7 us
6 KB current: standalone Ring LL 4.608 us, vLLM oneshot 4.000 us
```

The remaining difference comes from small implementation details:

| Item | Oneshot | Ring LL |
|---|---|---|
| Synchronization | global start/end barriers | LL flags and credit waits |
| Data movement | direct peer read | neighbor LL transfer |
| Critical path | barrier + one peer read | readLL flag/data wait |
| TP=2 expectation | very close | very close |

So if TP=2 standalone Ring LL is near oneshot, that is expected.

## Why Ring LL Gets Worse With More GPUs

Ring LL makes progress through ordered neighbor handoff:

```text
previous rank writes data+flag
current rank readLL spins until flag/data arrives
current rank reduces or forwards
current rank storeLL writes data+flag to next rank
```

For an all-reduce, the protocol has reduce-scatter plus allgather style work.
The primitive count grows with rank count:

```text
RS nranks steps + AG nranks - 1 steps = 2 * nranks - 1 primitive calls
```

The important part is that these steps are not all independent. A rank often
cannot proceed until the previous rank has produced the next LL packet. That is
the serialized latency chain.

clock64 phase attribution for standalone Ring LL at 6 KB:

| Phase | 2 GPU | 4 GPU | Meaning |
|---|---:|---:|---|
| total | 6.3 us | 9.7 us | rank 0 instrumented total |
| readLL spin | 4.3 us | 6.5 us | waiting for previous-rank flag/data |
| waitSend | 1.0 us | 1.2 us | credit/window wait |
| storeLL | 0.2 us | 0.2 us | posted data+flag stores |
| local load | 0.1 us | 0.1 us | local input load |
| reduce | ~0.0 us | ~0.0 us | arithmetic is negligible |
| other | 0.8 us | 1.6 us | loop/control overhead |

The bottleneck is `readLL spin`. That means Ring LL is waiting on serialized
communication visibility, not spending time on addition.

## Why Oneshot Can Be Faster

Oneshot uses a different latency shape:

```text
barrier_at_start
rank reads every peer input directly
local reduce
write output
barrier_at_end
```

After the start barrier, all ranks issue peer reads independently. There is no
requirement that data from rank 0 first pass through rank 1, then rank 2, then
rank 3. The peer reads can overlap.

For small messages, this matters more than bandwidth:

| Dimension | Oneshot | Ring LL |
|---|---|---|
| Dependency chain | shallow | neighbor-hop serialized |
| GPU count scaling | more peer reads, mostly parallel | more ordered protocol steps |
| Dominant wait | barrier/remote read fan-in | readLL spin |
| Best use case | small latency-sensitive messages | larger/general collective path |

This is why the custom oneshot mental model is:

```text
pay sync once, then pull everyone in parallel
```

and the Ring LL mental model is:

```text
pay many small ordered waits as packets move around the ring
```

## What `readLL spin` Means

`readLL spin` is the time a rank spends waiting for the previous rank's LL
packet to become visible. In LL protocol, the data and a small flag are coupled.
The consumer repeatedly checks the flag/data slot until it can safely consume
the packet.

Large `readLL spin` means:

```text
the current rank is ready
but the previous rank's packet has not arrived or become visible yet
```

On PCIe, each of those visibility waits has real latency. With more ranks, the
number of ordered waits increases.

## What `waitSend` Means

`waitSend` is the producer-side credit/window wait. It prevents a sender from
overrunning the receiver's LL buffers.

In our measurements it is much smaller than `readLL spin`:

```text
6 KB 2 GPU: readLL spin 4.3 us, waitSend 1.0 us
6 KB 4 GPU: readLL spin 6.5 us, waitSend 1.2 us
```

So the main Ring LL cost is not posted stores or credit waiting. It is waiting
for the previous rank's packet on the critical path.

## Why NCCL Ring LL Is Much Slower Than Standalone Ring LL

Standalone Ring LL is only the extracted protocol. NCCL is a production
collective runtime. It adds machinery that is valuable for correctness,
generality, topology handling, and integration, but it costs latency for tiny
messages.

Historical 1 KB result:

```text
Standalone Ring LL 4 GPU: 8.3 us
NCCL Ring LL 4 GPU:       17.2 us
```

We now also have an instrumented NCCL Ring LL counter run in
`nccl_ringll_internal_breakdown.md`. For the instrumented path, the largest
named LL primitive at 6 KB is `readLL`, while `waitSend` is not a bottleneck:

| Size | GPUs | readLL % | waitSend % | storeLL % | barrier % | other % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 KB | 2 | 6.13 | 1.02 | 0.45 | 1.38 | 91.02 |
| 1 KB | 4 | 3.98 | 1.27 | 0.11 | 1.76 | 92.89 |
| 6 KB | 2 | 33.81 | 0.82 | 2.38 | 1.18 | 61.80 |
| 6 KB | 4 | 20.35 | 1.12 | 0.59 | 1.54 | 76.41 |

These are aggregate thread-cycle percentages from an instrumented NCCL build,
not production wall time. The large `other` bucket includes generic loop/control
work, local load/reduce work, uninstrumented pieces, and instrumentation cost.

So the extra NCCL cost likely comes from:

```text
channel/work management
proxy/progress machinery
protocol setup and scheduling
more conservative generic paths
multi-channel/runtime bookkeeping
```

So there are two separate reasons for slowness:

```text
Ring protocol reason: readLL waits are serialized
NCCL implementation reason: generic/control work around the primitive
```

## Practical Takeaway

For TP=2, standalone Ring LL should be close to oneshot. If it is not, suspect
measurement method or implementation details first.

For 4 GPUs and tiny messages, Ring LL suffers because serialized `readLL` waits
grow with rank count. Oneshot can win because it converts the operation into one
sync plus parallel peer reads.

For full NCCL Ring LL, expect additional overhead over standalone Ring LL. That
gap is NCCL generic/control cost plus uninstrumented primitive work, not just
the LL protocol itself.
