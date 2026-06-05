/*
 * Standalone Ring LL AllReduce — extracted from NCCL v2.27.5.
 * Faithful reproduction of prims_ll.h + all_reduce.h runRing() for general nranks.
 *
 * Build:  nvcc -O3 -std=c++17 -arch=sm_120 standalone_ringll.cu -o standalone_ringll -lpthread
 * Run:    ./standalone_ringll [ngpus] [size_bytes] [iters] [warmup] [label|--label label]
 *         ./standalone_ringll 4 1024 100
 *         ./standalone_ringll              # default: detect GPUs, sweep sizes
 *
 * Copyright: Original NCCL code is (c) NVIDIA, BSD-3-Clause.
 * This file extracts and simplifies for standalone benchmarking only.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#define CHECK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// ============================================================
// Constants (from NCCL device.h / nccl_common.h)
// ============================================================
#define NCCL_STEPS 8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#define NCCL_LL_CLEAN_MASK 0x7ffffff8

#define MAX_GPUS 8

union ncclLLFifoLine {
  struct { uint32_t data1; uint32_t flag1; uint32_t data2; uint32_t flag2; };
  uint4 i4;
};

struct ConnInfo {
  ncclLLFifoLine* llBuff;
  volatile uint64_t* head;
  uint64_t step;
};

struct KernelArgs {
  float* sendbuff;
  float* recvbuff;
  ConnInfo sendConn;  // to next rank in ring
  ConnInfo recvConn;  // from prev rank in ring
  int rank;
  int nranks;
  int count;
};

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

const char* parse_label_arg(int argc, char** argv, int first_arg) {
  for (int i = first_arg; i < argc; i++) {
    if ((strcmp(argv[i], "--label") == 0 || strcmp(argv[i], "--lable") == 0) &&
        i + 1 < argc) {
      return argv[i + 1];
    }
    if (strncmp(argv[i], "--label=", 8) == 0) return argv[i] + 8;
    if (strncmp(argv[i], "--lable=", 8) == 0) return argv[i] + 8;
    if (strncmp(argv[i], "label=", 6) == 0) return argv[i] + 6;
    if (strncmp(argv[i], "lable=", 6) == 0) return argv[i] + 6;
    if (argv[i][0] != '-') return argv[i];
  }
  return nullptr;
}

// ============================================================
// LL primitives (from prims_ll.h)
// ============================================================
constexpr int EltPerLine = 2;
constexpr int ChunkAlign = 16 / sizeof(float);  // 4

#ifndef RINGLL_THREADS
#define RINGLL_THREADS 512
#endif

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

__global__ void initBuffers(float* sendbuf, float* recvbuf, int count, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    sendbuf[idx] = value;
    recvbuf[idx] = 0.0f;
  }
}

// ============================================================
// Kernel: Ring LL AllReduce for general nranks
// Follows NCCL all_reduce.h runRing() exactly.
//
// Ring: rank -> next=(rank+1)%nranks, prev=(rank+nranks-1)%nranks
//
// Reduce-scatter (nranks primitive calls):
//   step 0:            directSend(chunk[rank+nranks-1])
//   step j=2..nranks-1: directRecvReduceDirectSend(chunk[rank+nranks-j])
//   step nranks-1:      directRecvReduceCopyDirectSend(chunk[rank])
//
// All-gather (nranks-1 primitive calls):
//   step j=1..nranks-2: directRecvCopyDirectSend(chunk[rank+nranks-j])
//   final:              directRecv(chunk[rank+1])
// ============================================================

__global__ void __launch_bounds__(512)
ringLL_allreduce(KernelArgs args, int stepLines, Timing* timing) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int rank = args.rank;
  const int nranks = args.nranks;
  const int count = args.count;

  float* sendbuff = args.sendbuff;
  float* recvbuff = args.recvbuff;

  ncclLLFifoLine* sendBuff = args.sendConn.llBuff;
  ncclLLFifoLine* recvBuff = args.recvConn.llBuff;

  uint64_t sendStep = args.sendConn.step;
  uint64_t recvStep = args.recvConn.step;

  volatile uint64_t* sendHead = args.sendConn.head;
  volatile uint64_t* recvHead = args.recvConn.head;
  uint64_t sendHeadVal = sendStep;
  uint64_t sendHeadCache = (sendHead && tid == 0) ? *sendHead : 0;
  uint64_t recvHeadVal = recvStep;

  uint64_t t_total = clock64();
  uint64_t t_readll = 0, t_storell = 0, t_waitsend = 0, t_barrier = 0, t_load = 0, t_reduce = 0;
  int n_spins = 0;

  auto sendOffset = [&]() { return (int)(sendStep % NCCL_STEPS) * stepLines; };
  auto recvOffset = [&]() { return (int)(recvStep % NCCL_STEPS) * stepLines; };
  auto sendFlag   = [&]() { return NCCL_LL_FLAG(sendStep + 1); };
  auto recvFlag   = [&]() { return NCCL_LL_FLAG(recvStep + 1); };

  auto waitSend = [&]() {
    uint64_t tw = clock64();
    if (sendHead && tid == 0) {
      while (sendHeadCache + NCCL_STEPS < sendHeadVal + 1)
        sendHeadCache = *sendHead;
      sendHeadVal++;
    }
    __syncthreads();
    t_waitsend += clock64() - tw;
  };
  auto incRecv = [&]() { recvStep++; };
  auto postRecv = [&]() {
    uint64_t tb = clock64();
    __syncthreads();
    t_barrier += clock64() - tb;
    if (recvHead && tid == 0) *recvHead = ++recvHeadVal;
  };
  auto incSend = [&](int lastOffset) {
    if ((sendStep & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      int off = sendOffset();
      for (int o = lastOffset; o < stepLines; o += nthreads)
        doStoreLL(&sendBuff[off + o], 0, sendFlag());
    }
    sendStep++;
  };

  auto modRanks = [&](int r) -> int { return r >= nranks ? r - nranks : r; };

  int chunkCount = (count + nranks - 1) / nranks;
  chunkCount = ((chunkCount + ChunkAlign - 1) / ChunkAlign) * ChunkAlign;

  // Helper: get chunk offset & nelem
  auto chunkInfo = [&](int chunk, int& chunkOff, int& nelem) {
    chunkOff = chunk * chunkCount;
    nelem = count - chunkOff;
    if (nelem > chunkCount) nelem = chunkCount;
    if (nelem < 0) nelem = 0;
  };

  // ======== Reduce-Scatter ========

  // Step 0: directSend — push our chunk to next rank
  {
    int chunk = modRanks(rank + nranks - 1);
    int chunkOff, nelem;
    chunkInfo(chunk, chunkOff, nelem);
    int nlines = (nelem + EltPerLine - 1) / EltPerLine;

    waitSend();
    int off = sendOffset();
    uint32_t flag = sendFlag();
    int idx = tid;
    for (; idx < nlines; idx += nthreads) {
      int eidx = chunkOff + idx * EltPerLine;
      uint64_t tl = clock64();
      uint32_t d1 = (eidx < count) ? __float_as_uint(sendbuff[eidx]) : 0;
      uint32_t d2 = (eidx + 1 < count) ? __float_as_uint(sendbuff[eidx + 1]) : 0;
      t_load += clock64() - tl;
      uint64_t ts = clock64();
      doStoreLL(&sendBuff[off + idx], d1 | ((uint64_t)d2 << 32), flag);
      t_storell += clock64() - ts;
    }
    incSend(idx);
  }

  // Steps j=2..nranks-1: directRecvReduceDirectSend
  // recv from prev, reduce with local input, send to next (NO output write)
  for (int j = 2; j < nranks; ++j) {
    int chunk = modRanks(rank + nranks - j);
    int chunkOff, nelem;
    chunkInfo(chunk, chunkOff, nelem);
    int nlines = (nelem + EltPerLine - 1) / EltPerLine;

    waitSend();
    int soff = sendOffset();
    uint32_t sflag = sendFlag();
    int roff = recvOffset();
    uint32_t rflag = recvFlag();

    int idx = tid;
    for (; idx < nlines; idx += nthreads) {
      int eidx = chunkOff + idx * EltPerLine;
      uint64_t tl = clock64();
      uint32_t ld1 = (eidx < count) ? __float_as_uint(sendbuff[eidx]) : 0;
      uint32_t ld2 = (eidx + 1 < count) ? __float_as_uint(sendbuff[eidx + 1]) : 0;
      t_load += clock64() - tl;

      uint64_t pv = doReadLL(&recvBuff[roff + idx], rflag, &t_readll, &n_spins);

      uint64_t tr = clock64();
      float r1 = __uint_as_float(ld1) + __uint_as_float((uint32_t)pv);
      float r2 = __uint_as_float(ld2) + __uint_as_float((uint32_t)(pv >> 32));
      uint64_t rval = __float_as_uint(r1) | ((uint64_t)__float_as_uint(r2) << 32);
      t_reduce += clock64() - tr;

      uint64_t ts = clock64();
      doStoreLL(&sendBuff[soff + idx], rval, sflag);
      t_storell += clock64() - ts;
    }
    incRecv();
    postRecv();
    incSend(idx);
  }

  // Final RS step: directRecvReduceCopyDirectSend
  // recv + reduce + write output + send
  {
    int chunk = rank;
    int chunkOff, nelem;
    chunkInfo(chunk, chunkOff, nelem);
    int nlines = (nelem + EltPerLine - 1) / EltPerLine;

    waitSend();
    int soff = sendOffset();
    uint32_t sflag = sendFlag();
    int roff = recvOffset();
    uint32_t rflag = recvFlag();

    int idx = tid;
    for (; idx < nlines; idx += nthreads) {
      int eidx = chunkOff + idx * EltPerLine;
      uint64_t tl = clock64();
      uint32_t ld1 = (eidx < count) ? __float_as_uint(sendbuff[eidx]) : 0;
      uint32_t ld2 = (eidx + 1 < count) ? __float_as_uint(sendbuff[eidx + 1]) : 0;
      t_load += clock64() - tl;

      uint64_t pv = doReadLL(&recvBuff[roff + idx], rflag, &t_readll, &n_spins);

      uint64_t tr = clock64();
      float r1 = __uint_as_float(ld1) + __uint_as_float((uint32_t)pv);
      float r2 = __uint_as_float(ld2) + __uint_as_float((uint32_t)(pv >> 32));
      t_reduce += clock64() - tr;

      if (eidx < count) recvbuff[eidx] = r1;
      if (eidx + 1 < count) recvbuff[eidx + 1] = r2;

      uint64_t ts = clock64();
      uint64_t rval = __float_as_uint(r1) | ((uint64_t)__float_as_uint(r2) << 32);
      doStoreLL(&sendBuff[soff + idx], rval, sflag);
      t_storell += clock64() - ts;
    }
    incRecv();
    postRecv();
    incSend(idx);
  }

  // ======== All-Gather ========

  // Steps j=1..nranks-2: directRecvCopyDirectSend
  // recv + write output + forward to next
  for (int j = 1; j < nranks - 1; ++j) {
    int chunk = modRanks(rank + nranks - j);
    int chunkOff, nelem;
    chunkInfo(chunk, chunkOff, nelem);
    int nlines = (nelem + EltPerLine - 1) / EltPerLine;

    waitSend();
    int soff = sendOffset();
    uint32_t sflag = sendFlag();
    int roff = recvOffset();
    uint32_t rflag = recvFlag();

    int idx = tid;
    for (; idx < nlines; idx += nthreads) {
      int eidx = chunkOff + idx * EltPerLine;

      uint64_t pv = doReadLL(&recvBuff[roff + idx], rflag, &t_readll, &n_spins);
      uint32_t d1 = (uint32_t)pv, d2 = (uint32_t)(pv >> 32);

      if (eidx < count) recvbuff[eidx] = __uint_as_float(d1);
      if (eidx + 1 < count) recvbuff[eidx + 1] = __uint_as_float(d2);

      uint64_t ts = clock64();
      doStoreLL(&sendBuff[soff + idx], pv, sflag);
      t_storell += clock64() - ts;
    }
    incRecv();
    postRecv();
    incSend(idx);
  }

  // Final AG: directRecv — recv and write output, no send
  {
    int chunk = modRanks(rank + 1);
    int chunkOff, nelem;
    chunkInfo(chunk, chunkOff, nelem);
    int nlines = (nelem + EltPerLine - 1) / EltPerLine;

    int roff = recvOffset();
    uint32_t rflag = recvFlag();

    for (int idx = tid; idx < nlines; idx += nthreads) {
      int eidx = chunkOff + idx * EltPerLine;
      uint64_t pv = doReadLL(&recvBuff[roff + idx], rflag, &t_readll, &n_spins);
      if (eidx < count) recvbuff[eidx] = __uint_as_float((uint32_t)pv);
      if (eidx + 1 < count) recvbuff[eidx + 1] = __uint_as_float((uint32_t)(pv >> 32));
    }
    incRecv();
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
  ncclLLFifoLine* llBuff;
  uint64_t* headCounter;
  Timing* timing;
  cudaStream_t stream;
  cudaEvent_t ev_start, ev_stop;
};

struct BenchArgs {
  int rank;
  int nranks;
  size_t bytes;
  int iters;
  int warmup;
  GPUState* states;
  pthread_barrier_t* bar;
  int stepLines;
  int llBufSize;
  float median_us;
  Timing avg_timing;
};

void* bench_thread(void* arg) {
  BenchArgs* a = (BenchArgs*)arg;
  int rank = a->rank;
  int nranks = a->nranks;
  int next = (rank + 1) % nranks;
  CHECK(cudaSetDevice(rank));

  auto& s = a->states[rank];
  auto& ns = a->states[next];  // next rank in ring
  int count = a->bytes / sizeof(float);
  int stepLines = a->stepLines;
  int llBufSize = a->llBufSize;

  // Reallocate LL buffer
  CHECK(cudaFree(s.llBuff));
  CHECK(cudaMalloc(&s.llBuff, llBufSize));
  CHECK(cudaMemsetAsync(s.llBuff, 0, llBufSize, s.stream));
  CHECK(cudaMemsetAsync(s.headCounter, 0, sizeof(uint64_t), s.stream));
  CHECK(cudaStreamSynchronize(s.stream));
  pthread_barrier_wait(a->bar);  // ensure all GPUs done alloc before capturing pointers

  KernelArgs kargs;
  kargs.sendbuff = s.sendbuf;
  kargs.recvbuff = s.recvbuf;
  kargs.rank = rank;
  kargs.nranks = nranks;
  kargs.count = count;

  // Ring: send to next, recv from prev
  // sendConn.llBuff = next rank's receive buffer (we write there)
  kargs.sendConn.llBuff = ns.llBuff;
  kargs.sendConn.head = ns.headCounter;
  kargs.sendConn.step = 0;

  // recvConn.llBuff = our own receive buffer (prev writes here)
  kargs.recvConn.llBuff = s.llBuff;
  kargs.recvConn.head = s.headCounter;
  kargs.recvConn.step = 0;

  // Init input data: rank i gets value (i+1)
  int initBlocks = (count + 255) / 256;
  if (initBlocks > 0) {
    initBuffers<<<initBlocks, 256, 0, s.stream>>>(s.sendbuf, s.recvbuf, count, float(rank + 1));
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(s.stream));
  }
  pthread_barrier_wait(a->bar);

  // Warmup
  for (int w = 0; w < a->warmup; w++) {
    CHECK(cudaMemsetAsync(s.llBuff, 0, llBufSize, s.stream));
    CHECK(cudaMemsetAsync(s.headCounter, 0, sizeof(uint64_t), s.stream));
    CHECK(cudaStreamSynchronize(s.stream));
    kargs.sendConn.step = 0;
    kargs.recvConn.step = 0;
    pthread_barrier_wait(a->bar);

    ringLL_allreduce<<<1, RINGLL_THREADS, 0, s.stream>>>(kargs, stepLines, nullptr);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(s.stream));
    pthread_barrier_wait(a->bar);
  }
  pthread_barrier_wait(a->bar);

  // Benchmark
  std::vector<float> times(a->iters);
  Timing total_timing = {};
  for (int it = 0; it < a->iters; it++) {
    CHECK(cudaMemsetAsync(s.llBuff, 0, llBufSize, s.stream));
    CHECK(cudaMemsetAsync(s.headCounter, 0, sizeof(uint64_t), s.stream));
    CHECK(cudaStreamSynchronize(s.stream));
    kargs.sendConn.step = 0;
    kargs.recvConn.step = 0;
    pthread_barrier_wait(a->bar);

    CHECK(cudaEventRecord(s.ev_start, s.stream));
    ringLL_allreduce<<<1, RINGLL_THREADS, 0, s.stream>>>(kargs, stepLines, s.timing);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(s.ev_stop, s.stream));
    CHECK(cudaStreamSynchronize(s.stream));
    pthread_barrier_wait(a->bar);

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

  std::sort(times.begin(), times.end());
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
  int ngpus_available;
  CHECK(cudaGetDeviceCount(&ngpus_available));

  int ngpus = argc > 1 ? atoi(argv[1]) : ngpus_available;
  if (ngpus < 2 || ngpus > MAX_GPUS || ngpus > ngpus_available) {
    fprintf(stderr, "ngpus must be 2..%d (available: %d)\n",
            MAX_GPUS, ngpus_available);
    return 1;
  }

  size_t sizes_default[] = {1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
  size_t single_size = argc > 2 ? atoi(argv[2]) : 0;
  int iters = argc > 3 ? atoi(argv[3]) : 100;
  int warmup = argc > 4 ? atoi(argv[4]) : 20;
  const char* label = parse_label_arg(argc, argv, 5);
  if (label && label[0] == '\0') label = nullptr;
  if (iters <= 0) { fprintf(stderr, "iters must be positive\n"); return 1; }

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

  GPUState states[MAX_GPUS];
  size_t max_bytes = sizes[nsizes - 1];
  for (int i = 0; i < ngpus; i++) {
    CHECK(cudaSetDevice(i));
    for (int j = 0; j < ngpus; j++) {
      if (i == j) continue;
      int c;
      CHECK(cudaDeviceCanAccessPeer(&c, i, j));
      if (c) CHECK(cudaDeviceEnablePeerAccess(j, 0));
    }
    CHECK(cudaMalloc(&states[i].sendbuf, max_bytes));
    CHECK(cudaMalloc(&states[i].recvbuf, max_bytes));
    CHECK(cudaMalloc(&states[i].llBuff, 1024));  // resized in bench_thread
    CHECK(cudaMalloc(&states[i].headCounter, sizeof(uint64_t)));
    CHECK(cudaMalloc(&states[i].timing, sizeof(Timing)));
    CHECK(cudaStreamCreate(&states[i].stream));
    CHECK(cudaEventCreate(&states[i].ev_start));
    CHECK(cudaEventCreate(&states[i].ev_stop));
  }

  // Expected sum = 1 + 2 + ... + ngpus = ngpus*(ngpus+1)/2
  float expected = (float)(ngpus * (ngpus + 1) / 2);

  printf("Standalone Ring LL AllReduce (extracted from NCCL v2.27.5)\n");
  printf("  %d GPUs, ring: 0", ngpus);
  for (int i = 1; i < ngpus; i++) printf("->%d", i);
  printf("->0, clock=%.0f MHz, warmup=%d, iters=%d, threads=%d, label=%s\n",
         cyc_per_us, warmup, iters, RINGLL_THREADS, label ? label : "-");
  printf("  expected sum = %.0f\n\n", expected);

  bool all_verify_ok = true;
  for (int si = 0; si < nsizes; si++) {
    size_t bytes = sizes[si];
    int count = bytes / sizeof(float);

    // LL buffer sizing
    int chunkCount = ((count + ngpus - 1) / ngpus + ChunkAlign - 1) / ChunkAlign * ChunkAlign;
    int linesNeeded = (chunkCount + EltPerLine - 1) / EltPerLine;
    int stepLines = std::max(linesNeeded + 32, 1024);
    int llBufSize = NCCL_STEPS * stepLines * (int)sizeof(ncclLLFifoLine);

    pthread_barrier_t bar;
    pthread_barrier_init(&bar, nullptr, ngpus);
    std::vector<BenchArgs> args(ngpus);
    std::vector<pthread_t> threads(ngpus);
    for (int r = 0; r < ngpus; r++) {
      args[r] = {r, ngpus, bytes, iters, warmup, states, &bar, stepLines, llBufSize, 0, {}};
      pthread_create(&threads[r], nullptr, bench_thread, &args[r]);
    }
    for (int r = 0; r < ngpus; r++) pthread_join(threads[r], nullptr);
    pthread_barrier_destroy(&bar);

    // Verify
    bool verify_ok = true;
    int bad_rank = -1, bad_index = -1;
    float bad_value = 0.0f;
    for (int r = 0; r < ngpus && verify_ok; r++) {
      CHECK(cudaSetDevice(r));
      std::vector<float> host(count);
      if (count > 0)
        CHECK(cudaMemcpy(host.data(), states[r].recvbuf, count * sizeof(float), cudaMemcpyDeviceToHost));
      for (int i = 0; i < count; i++) {
        if (fabsf(host[i] - expected) > 1e-3f) {
          verify_ok = false;
          bad_rank = r;
          bad_index = i;
          bad_value = host[i];
          break;
        }
      }
    }
    all_verify_ok = all_verify_ok && verify_ok;

    // Report: max across all ranks
    float max_us = 0;
    for (int r = 0; r < ngpus; r++)
      max_us = std::max(max_us, args[r].median_us);

    auto& t = args[0].avg_timing;
    float total_us = t.total / cyc_per_us;

    const char* unit = "B"; float ds = bytes;
    if (bytes >= 1024*1024) { ds = bytes/(1024.0*1024.0); unit = "MB"; }
    else if (bytes >= 1024) { ds = bytes/1024.0; unit = "KB"; }

    if (nsizes > 1) {
      printf("label=%s  %6.0f %-2s  event=%.1fus  clock=%.1fus  readLL=%.1f  storeLL=%.1f  "
             "waitSend=%.1f  barrier=%.1f  load=%.1f  reduce=%.1f  spins=%d  verify=%s\n",
             label ? label : "-", ds, unit, max_us, total_us,
             t.readll_spin / cyc_per_us, t.storell / cyc_per_us,
             t.waitsend / cyc_per_us, t.barrier / cyc_per_us,
             t.dataload / cyc_per_us, t.reduce / cyc_per_us,
             t.readll_spins, verify_ok ? "OK" : "FAIL");
    } else {
      printf("Size: %.0f %s\n", ds, unit);
      printf("Label: %s\n", label ? label : "-");
      if (verify_ok) {
        printf("Verification: OK (expected %.0f on all %d GPUs)\n", expected, ngpus);
      } else {
        printf("Verification: FAIL (rank=%d index=%d got=%g expected=%.0f)\n",
               bad_rank, bad_index, bad_value, expected);
      }
      printf("cudaEvent median: %.1f us (max across %d ranks)\n", max_us, ngpus);
      printf("clock64() total:  %.1f us (rank 0)\n\n", total_us);
      printf("%-25s %10s %10s\n", "Phase", "cycles", "us");
      printf("───────────────────────────────────────────────\n");
      printf("%-25s %10llu %10.1f\n", "readLL spin (flag wait)", (unsigned long long)t.readll_spin, t.readll_spin / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "storeLL (write d+f)", (unsigned long long)t.storell, t.storell / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "waitSend (credits)", (unsigned long long)t.waitsend, t.waitsend / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "__syncthreads", (unsigned long long)t.barrier, t.barrier / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "data load (local)", (unsigned long long)t.dataload, t.dataload / cyc_per_us);
      printf("%-25s %10llu %10.1f\n", "reduce (float sum)", (unsigned long long)t.reduce, t.reduce / cyc_per_us);
      uint64_t accounted = t.readll_spin + t.storell + t.waitsend + t.barrier + t.dataload + t.reduce;
      uint64_t other = t.total > accounted ? t.total - accounted : 0;
      printf("%-25s %10llu %10.1f\n", "other (overhead)", (unsigned long long)other, other / cyc_per_us);
      printf("───────────────────────────────────────────────\n");
      printf("%-25s %10llu %10.1f\n", "TOTAL", (unsigned long long)t.total, total_us);
      printf("\nAvg readLL spin iters: %d\n", t.readll_spins);
    }
  }

  for (int i = 0; i < ngpus; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(states[i].sendbuf)); CHECK(cudaFree(states[i].recvbuf));
    CHECK(cudaFree(states[i].llBuff)); CHECK(cudaFree(states[i].headCounter));
    CHECK(cudaFree(states[i].timing));
  }
  return all_verify_ok ? 0 : 2;
}
