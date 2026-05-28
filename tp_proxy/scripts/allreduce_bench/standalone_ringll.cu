/*
 * Standalone Ring LL AllReduce — extracted from NCCL v2.27.5.
 * Faithful reproduction of prims_ll.h + all_reduce.h runRing() for TP=2.
 *
 * Build:  nvcc -O3 -std=c++17 -arch=sm_120 standalone_ringll.cu -o standalone_ringll -lpthread
 * Run:    ./standalone_ringll [size_bytes] [iters]
 *
 * Copyright: Original NCCL code is (c) NVIDIA, BSD-3-Clause.
 * This file extracts and simplifies for standalone benchmarking only.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#define CHECK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// ============================================================
// Constants (from NCCL device.h / nccl_common.h)
// ============================================================
#define NCCL_STEPS 8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#define NCCL_LL_CLEAN_MASK 0x7ffffff8

union ncclLLFifoLine {
  struct { uint32_t data1; uint32_t flag1; uint32_t data2; uint32_t flag2; };
  uint4 i4;
};

// Simplified connection info for standalone use
struct ConnInfo {
  ncclLLFifoLine* llBuff;   // LL FIFO buffer (remote for send, local for recv)
  volatile uint64_t* head;  // credits: local for send, remote for recv
  volatile uint64_t* tail;  // Not used in LL, but keep for structure
  uint64_t step;
};

// Per-GPU state passed to kernel
struct KernelArgs {
  float* sendbuff;
  float* recvbuff;
  ConnInfo sendConn;  // connection to peer (we send to peer's llBuff)
  ConnInfo recvConn;  // connection from peer (peer sends to our llBuff)
  int rank;
  int nranks;
  int count;          // number of float elements
};

// Timing (written by thread 0)
struct Timing {
  uint64_t total;
  uint64_t readll_spin;
  uint64_t storell;
  uint64_t waitsend;
  uint64_t barrier;
  uint64_t dataload;
  uint64_t reduce;
  int readll_spins;
};

// ============================================================
// Kernel: Ring LL AllReduce (TP=2, 1 channel, float sum)
// Follows NCCL's prims_ll.h + all_reduce.h runRing() exactly.
// ============================================================

// --- LL primitives (from prims_ll.h) ---

__device__ __forceinline__ uint64_t
doReadLL(volatile ncclLLFifoLine* src, uint32_t flag, uint64_t* spin_cycles, int* spin_n) {
  uint32_t d1, f1, d2, f2;
  uint64_t t0 = clock64();
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d1), "=r"(f1), "=r"(d2), "=r"(f2)
                 : "l"(&src->i4) : "memory");
    (*spin_n)++;
  } while (f1 != flag || f2 != flag);
  *spin_cycles += clock64() - t0;
  return d1 | ((uint64_t)d2 << 32);
}

__device__ __forceinline__ void
doStoreLL(volatile ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(&dst->i4),
                  "r"((uint32_t)val), "r"(flag),
                  "r"((uint32_t)(val >> 32)), "r"(flag) : "memory");
}

// EltPerLine: 2 floats (8 bytes) per LL line
constexpr int EltPerLine = 2;

// --- The kernel ---
__global__ void __launch_bounds__(512)
ringLL_allreduce(KernelArgs args, int stepLines, Timing* timing) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int rank = args.rank;
  const int nranks = args.nranks;  // 2
  const int count = args.count;    // float elements

  float* sendbuff = args.sendbuff;
  float* recvbuff = args.recvbuff;

  // LL FIFO pointers
  ncclLLFifoLine* sendBuff = args.sendConn.llBuff;  // peer's recv FIFO
  ncclLLFifoLine* recvBuff = args.recvConn.llBuff;  // our recv FIFO

  // Step counters
  uint64_t sendStep = args.sendConn.step;
  uint64_t recvStep = args.recvConn.step;

  // Credit counters
  volatile uint64_t* sendHead = args.sendConn.head;  // recv side updates this
  volatile uint64_t* recvHead = args.recvConn.head;   // we update this
  uint64_t sendHeadCache = sendHead ? *sendHead : 0;
  uint64_t recvHeadVal = recvStep;

  // Timing accumulators (thread 0 only)
  uint64_t t_total = clock64();
  uint64_t t_readll = 0, t_storell = 0, t_waitsend = 0, t_barrier = 0, t_load = 0, t_reduce = 0;
  int n_spins = 0;

  // Helper lambdas
  auto sendOffset = [&]() { return (int)(sendStep % NCCL_STEPS) * stepLines; };
  auto recvOffset = [&]() { return (int)(recvStep % NCCL_STEPS) * stepLines; };
  auto sendFlag   = [&]() { return NCCL_LL_FLAG(sendStep + 1); };
  auto recvFlag   = [&]() { return NCCL_LL_FLAG(recvStep + 1); };

  // waitSend: check credits (from prims_ll.h:56-70)
  auto waitSend = [&]() {
    uint64_t tw = clock64();
    if (sendHead) {
      while (sendHeadCache + NCCL_STEPS < sendStep + 1) {
        sendHeadCache = *sendHead;
      }
      sendStep++;
    }
    __syncthreads();
    t_waitsend += clock64() - tw;
  };

  // postRecv: update recv head (from prims_ll.h:75-78)
  auto postRecv = [&]() {
    uint64_t tb = clock64();
    __syncthreads();
    t_barrier += clock64() - tb;
    if (recvHead && tid == 0) *recvHead = ++recvHeadVal;
    recvStep++;
  };

  // incSend: advance send step + cleanup (from prims_ll.h:80-87)
  auto incSend = [&](int lastOffset) {
    if ((sendStep & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      int off = sendOffset();
      for (int o = lastOffset; o < stepLines; o += nthreads)
        doStoreLL(&sendBuff[off + o], 0, sendFlag());
    }
    sendStep++;
  };

  // ============================================================
  // Ring AllReduce for nranks=2 (from all_reduce.h runRing)
  //
  // ringIx = rank (0 or 1)
  // prev = (rank + nranks - 1) % nranks = 1 - rank
  // next = (rank + 1) % nranks = 1 - rank
  //
  // chunkCount ≈ count / nranks (aligned)
  // For nranks=2:
  //   Step 0: send(chunk1)  — push my chunk to peer
  //   Step 1: recvReduceCopySend(chunk0) — recv + reduce + write + push
  //   Step 2: recv(chunk1) — recv final from peer
  // ============================================================

  int chunkCount = (count + nranks - 1) / nranks;
  // Align to EltPerLine
  chunkCount = ((chunkCount + EltPerLine - 1) / EltPerLine) * EltPerLine;

  int chunk0_offset, chunk1_offset, chunk0_n, chunk1_n;
  if (rank == 0) {
    // ringIx=0: step0 sends chunk1 (index=1), step1 processes chunk0 (index=0)
    chunk1_offset = 1 * chunkCount;
    chunk0_offset = 0 * chunkCount;
  } else {
    // ringIx=1: step0 sends chunk0 (index=0), step1 processes chunk1 (index=1)
    chunk1_offset = 0 * chunkCount;
    chunk0_offset = 1 * chunkCount;
  }
  chunk0_n = min(chunkCount, count - chunk0_offset);
  chunk1_n = min(chunkCount, count - chunk1_offset);
  if (chunk0_n < 0) chunk0_n = 0;
  if (chunk1_n < 0) chunk1_n = 0;

  // ---- Step 0: send(chunk1) — LLGenericOp<0,1,Input,-1> ----
  // Only SEND, read from sendbuff, write to peer's LL FIFO
  {
    waitSend();
    int off = sendOffset();
    uint32_t flag = sendFlag();
    float* src = sendbuff + chunk1_offset;
    int nelem = chunk1_n;

    for (int idx = tid; idx < (nelem + EltPerLine - 1) / EltPerLine; idx += nthreads) {
      int eidx = chunk1_offset + idx * EltPerLine;
      uint64_t tl = clock64();
      uint32_t d1 = (eidx < count) ? __float_as_uint(sendbuff[eidx]) : 0;
      uint32_t d2 = (eidx + 1 < count) ? __float_as_uint(sendbuff[eidx + 1]) : 0;
      t_load += clock64() - tl;
      uint64_t val = d1 | ((uint64_t)d2 << 32);

      uint64_t ts = clock64();
      doStoreLL(&sendBuff[off + idx], val, flag);
      t_storell += clock64() - ts;
    }
    incSend((chunk1_n + EltPerLine - 1) / EltPerLine);
  }

  // ---- Step 1: recvReduceCopySend(chunk0) — LLGenericOp<1,1,Input,Output> ----
  // RECV from peer + reduce with local + write output + SEND to peer
  {
    waitSend();
    int soff = sendOffset();
    uint32_t sflag = sendFlag();
    int roff = recvOffset();
    uint32_t rflag = recvFlag();

    for (int idx = tid; idx < (chunk0_n + EltPerLine - 1) / EltPerLine; idx += nthreads) {
      int eidx = chunk0_offset + idx * EltPerLine;

      // Load local data
      uint64_t tl = clock64();
      uint32_t local_d1 = (eidx < count) ? __float_as_uint(sendbuff[eidx]) : 0;
      uint32_t local_d2 = (eidx + 1 < count) ? __float_as_uint(sendbuff[eidx + 1]) : 0;
      t_load += clock64() - tl;

      // Read peer data from LL FIFO (spin on flag)
      uint64_t peerVal = doReadLL(&recvBuff[roff + idx], rflag, &t_readll, &n_spins);
      uint32_t peer_d1 = (uint32_t)peerVal;
      uint32_t peer_d2 = (uint32_t)(peerVal >> 32);

      // Reduce (float sum)
      uint64_t tr = clock64();
      float r1 = __uint_as_float(local_d1) + __uint_as_float(peer_d1);
      float r2 = __uint_as_float(local_d2) + __uint_as_float(peer_d2);
      uint32_t rd1 = __float_as_uint(r1);
      uint32_t rd2 = __float_as_uint(r2);
      t_reduce += clock64() - tr;

      // Write to output
      if (eidx < count) recvbuff[eidx] = r1;
      if (eidx + 1 < count) recvbuff[eidx + 1] = r2;

      // Send reduced result to peer
      uint64_t ts = clock64();
      uint64_t rval = rd1 | ((uint64_t)rd2 << 32);
      doStoreLL(&sendBuff[soff + idx], rval, sflag);
      t_storell += clock64() - ts;
    }
    postRecv();
    incSend((chunk0_n + EltPerLine - 1) / EltPerLine);
  }

  // ---- Step 2: recv(chunk1) — LLGenericOp<1,0,-1,Output> ----
  // RECV final result from peer, write to output
  {
    int roff = recvOffset();
    uint32_t rflag = recvFlag();

    for (int idx = tid; idx < (chunk1_n + EltPerLine - 1) / EltPerLine; idx += nthreads) {
      int eidx = chunk1_offset + idx * EltPerLine;

      uint64_t peerVal = doReadLL(&recvBuff[roff + idx], rflag, &t_readll, &n_spins);
      uint32_t d1 = (uint32_t)peerVal;
      uint32_t d2 = (uint32_t)(peerVal >> 32);

      if (eidx < count) recvbuff[eidx] = __uint_as_float(d1);
      if (eidx + 1 < count) recvbuff[eidx + 1] = __uint_as_float(d2);
    }
    postRecv();
  }

  t_total = clock64() - t_total;
  if (tid == 0 && timing) {
    timing->total = t_total;
    timing->readll_spin = t_readll;
    timing->storell = t_storell;
    timing->waitsend = t_waitsend;
    timing->barrier = t_barrier;
    timing->dataload = t_load;
    timing->reduce = t_reduce;
    timing->readll_spins = n_spins;
  }
}

// ============================================================
// Host benchmark
// ============================================================
struct GPUState {
  float* sendbuf;
  float* recvbuf;
  ncclLLFifoLine* llBuff;  // LL FIFO (peer writes to this)
  uint64_t* headCounter;   // credit counter
  Timing* timing;
  cudaStream_t stream;
  cudaEvent_t ev_start, ev_stop;
};

struct BenchArgs {
  int rank;
  size_t bytes;
  int iters;
  GPUState* states;
  pthread_barrier_t* bar;
  float median_us;
  Timing avg_timing;
};

void* bench_thread(void* arg) {
  BenchArgs* a = (BenchArgs*)arg;
  int rank = a->rank;
  int peer = 1 - rank;
  CHECK(cudaSetDevice(rank));

  auto& s = a->states[rank];
  auto& ps = a->states[peer];
  int count = a->bytes / sizeof(float);

  // LL buffer sizing: same as NCCL
  // stepLines = bufSize / NCCL_STEPS / sizeof(ncclLLFifoLine)
  // For small messages: need at least count/EltPerLine lines per step
  int linesNeeded = (count / 2 + EltPerLine - 1) / EltPerLine;  // per chunk (half the data)
  int stepLines = std::max(linesNeeded + 32, 1024);  // add margin
  int llBufSize = NCCL_STEPS * stepLines * sizeof(ncclLLFifoLine);

  // Reallocate LL buffer if needed
  CHECK(cudaFree(s.llBuff));
  CHECK(cudaMalloc(&s.llBuff, llBufSize));
  CHECK(cudaMemset(s.llBuff, 0, llBufSize));
  CHECK(cudaMemset(s.headCounter, 0, sizeof(uint64_t)));

  KernelArgs kargs;
  kargs.sendbuff = s.sendbuf;
  kargs.recvbuff = s.recvbuf;
  kargs.rank = rank;
  kargs.nranks = 2;
  kargs.count = count;

  // Send connection: our storeLL goes to PEER's llBuff
  kargs.sendConn.llBuff = ps.llBuff;
  kargs.sendConn.head = ps.headCounter;  // peer's head = our credit source
  kargs.sendConn.step = 0;

  // Recv connection: peer's storeLL goes to OUR llBuff
  kargs.recvConn.llBuff = s.llBuff;
  kargs.recvConn.head = s.headCounter;   // we update our own head
  kargs.recvConn.step = 0;

  CHECK(cudaMemset(s.sendbuf, rank + 1, a->bytes));
  pthread_barrier_wait(a->bar);

  // Warmup
  for (int w = 0; w < 20; w++) {
    // Reset LL buffer and counters each iteration
    CHECK(cudaMemset(s.llBuff, 0, llBufSize));
    CHECK(cudaMemset(s.headCounter, 0, sizeof(uint64_t)));
    kargs.sendConn.step = w * 3;  // 3 steps per allreduce
    kargs.recvConn.step = w * 3;
    pthread_barrier_wait(a->bar);

    ringLL_allreduce<<<1, 128, 0, s.stream>>>(kargs, stepLines, nullptr);
    CHECK(cudaStreamSynchronize(s.stream));
  }
  pthread_barrier_wait(a->bar);

  // Benchmark
  float times[4096];
  Timing total_timing = {};
  for (int it = 0; it < a->iters; it++) {
    CHECK(cudaMemset(s.llBuff, 0, llBufSize));
    CHECK(cudaMemset(s.headCounter, 0, sizeof(uint64_t)));
    kargs.sendConn.step = (20 + it) * 3;
    kargs.recvConn.step = (20 + it) * 3;
    pthread_barrier_wait(a->bar);

    CHECK(cudaEventRecord(s.ev_start, s.stream));
    ringLL_allreduce<<<1, 128, 0, s.stream>>>(kargs, stepLines, s.timing);
    CHECK(cudaEventRecord(s.ev_stop, s.stream));
    CHECK(cudaStreamSynchronize(s.stream));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, s.ev_start, s.ev_stop));
    times[it] = ms * 1000;

    Timing t;
    CHECK(cudaMemcpy(&t, s.timing, sizeof(Timing), cudaMemcpyDeviceToHost));
    total_timing.total += t.total;
    total_timing.readll_spin += t.readll_spin;
    total_timing.storell += t.storell;
    total_timing.waitsend += t.waitsend;
    total_timing.barrier += t.barrier;
    total_timing.dataload += t.dataload;
    total_timing.reduce += t.reduce;
    total_timing.readll_spins += t.readll_spins;
  }

  // Sort, median
  std::sort(times, times + a->iters);
  a->median_us = times[a->iters / 2];

  auto& at = a->avg_timing;
  int n = a->iters;
  at.total = total_timing.total / n;
  at.readll_spin = total_timing.readll_spin / n;
  at.storell = total_timing.storell / n;
  at.waitsend = total_timing.waitsend / n;
  at.barrier = total_timing.barrier / n;
  at.dataload = total_timing.dataload / n;
  at.reduce = total_timing.reduce / n;
  at.readll_spins = total_timing.readll_spins / n;
  return nullptr;
}

int main(int argc, char** argv) {
  size_t sizes_default[] = {1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
  size_t single_size = argc > 1 ? atoi(argv[1]) : 0;
  int iters = argc > 2 ? atoi(argv[2]) : 100;

  size_t* sizes;
  int nsizes;
  if (single_size > 0) {
    sizes = &single_size;
    nsizes = 1;
  } else {
    sizes = sizes_default;
    nsizes = 10;
  }

  int clock_khz;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  float cyc_per_us = clock_khz / 1000.0f;

  GPUState states[2];
  size_t max_bytes = sizes[nsizes - 1];
  for (int i = 0; i < 2; i++) {
    CHECK(cudaSetDevice(i));
    int c; CHECK(cudaDeviceCanAccessPeer(&c, i, 1-i)); if(c) CHECK(cudaDeviceEnablePeerAccess(1-i, 0));
    CHECK(cudaMalloc(&states[i].sendbuf, max_bytes));
    CHECK(cudaMalloc(&states[i].recvbuf, max_bytes));
    CHECK(cudaMalloc(&states[i].llBuff, 1024));  // resized in bench_thread
    CHECK(cudaMalloc(&states[i].headCounter, sizeof(uint64_t)));
    CHECK(cudaMalloc(&states[i].timing, sizeof(Timing)));
    CHECK(cudaStreamCreate(&states[i].stream));
    CHECK(cudaEventCreate(&states[i].ev_start));
    CHECK(cudaEventCreate(&states[i].ev_stop));
  }

  printf("Standalone Ring LL AllReduce (extracted from NCCL v2.27.5)\n");
  printf("  2 GPUs, clock=%.0f MHz, iters=%d\n\n", cyc_per_us, iters);

  for (int si = 0; si < nsizes; si++) {
    size_t bytes = sizes[si];

    pthread_barrier_t bar;
    pthread_barrier_init(&bar, nullptr, 2);
    BenchArgs args[2];
    pthread_t threads[2];
    for (int r = 0; r < 2; r++) {
      args[r] = {r, bytes, iters, states, &bar, 0, {}};
      pthread_create(&threads[r], nullptr, bench_thread, &args[r]);
    }
    for (int r = 0; r < 2; r++) pthread_join(threads[r], nullptr);
    pthread_barrier_destroy(&bar);

    float max_us = std::max(args[0].median_us, args[1].median_us);
    auto& t = args[0].avg_timing;
    float total_us = t.total / cyc_per_us;

    const char* unit = "B"; float ds = bytes;
    if (bytes >= 1024*1024) { ds = bytes/(1024.0*1024.0); unit = "MB"; }
    else if (bytes >= 1024) { ds = bytes/1024.0; unit = "KB"; }

    if (nsizes > 1) {
      printf("%6.0f %-2s  event=%.1fus  clock=%.1fus  readLL=%.1f  storeLL=%.1f  "
             "waitSend=%.1f  barrier=%.1f  load=%.1f  reduce=%.1f  spins=%d\n",
             ds, unit, max_us, total_us,
             t.readll_spin / cyc_per_us, t.storell / cyc_per_us,
             t.waitsend / cyc_per_us, t.barrier / cyc_per_us,
             t.dataload / cyc_per_us, t.reduce / cyc_per_us,
             t.readll_spins);
    } else {
      printf("Size: %.0f %s\n", ds, unit);
      printf("cudaEvent median: %.1f us\n", max_us);
      printf("clock64() total:  %.1f us\n\n", total_us);
      printf("%-25s %10s %10s\n", "Phase", "cycles", "us");
      printf("───────────────────────────────────────────────\n");
      printf("%-25s %10llu %10.1f\n", "readLL spin (flag wait)", t.readll_spin, t.readll_spin / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "storeLL (write d+f)", t.storell, t.storell / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "waitSend (credits)", t.waitsend, t.waitsend / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "__syncthreads", t.barrier, t.barrier / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "data load (local)", t.dataload, t.dataload / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "reduce (float sum)", t.reduce, t.reduce / cyc_per_us);
      uint64_t accounted = t.readll_spin + t.storell + t.waitsend + t.barrier + t.dataload + t.reduce;
      uint64_t other = t.total > accounted ? t.total - accounted : 0;
      printf("%-25s %10llu %10.1f\n", "other (overhead)", other, other / cyc_per_us);
      printf("───────────────────────────────────────────────\n");
      printf("%-25s %10llu %10.1f\n", "TOTAL", t.total, total_us);
      printf("\nAvg readLL spin iters: %d\n", t.readll_spins);
    }
  }

  for (int i = 0; i < 2; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(states[i].sendbuf)); CHECK(cudaFree(states[i].recvbuf));
    CHECK(cudaFree(states[i].llBuff)); CHECK(cudaFree(states[i].headCounter));
    CHECK(cudaFree(states[i].timing));
  }
  return 0;
}
