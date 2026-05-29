/*
 * Micro-benchmarks to isolate WHY oneshot is faster than Ring LL.
 * Tests each factor independently:
 *
 * 1. barrier_only     - Pure synchronization cost (no data)
 * 2. read_peer        - PCIe read from peer (no sync)
 * 3. write_peer       - PCIe write to peer (no sync, like Ring push)
 * 4. write_with_flag  - PCIe write + LL-style flag per 8B (Ring LL simulation)
 * 5. read_with_barrier- Oneshot: barrier + bulk read + barrier
 * 6. write_two_phase  - Ring-style: 2-phase write (reduce-scatter + all-gather)
 *
 * Build: nvcc -O3 -std=c++17 -arch=sm_120 bench_breakdown.cu -o bench_breakdown -lpthread
 * Run:   ./bench_breakdown [ngpus] [size_bytes]
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

using FlagType = uint32_t;
constexpr int kMaxBlocks = 36;

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
};

struct __align__(16) RankSignals { Signal* signals[8]; };

// --- Barrier (same as oneshot) ---
static __device__ __forceinline__ void st_vol(FlagType* a, FlagType f) {
  asm volatile("st.volatile.global.u32 [%1], %0;"::"r"(f),"l"(a));
}
static __device__ __forceinline__ FlagType ld_vol(FlagType* a) {
  FlagType f; asm volatile("ld.volatile.global.u32 %0, [%1];":"=r"(f):"l"(a)); return f;
}
static __device__ __forceinline__ void st_rel(FlagType* a, FlagType f) {
  asm volatile("st.release.sys.global.u32 [%1], %0;"::"r"(f),"l"(a));
}
static __device__ __forceinline__ FlagType ld_acq(FlagType* a) {
  FlagType f; asm volatile("ld.acquire.sys.global.u32 %0, [%1];":"=r"(f):"l"(a)); return f;
}

template <int ngpus, bool need_fence>
__device__ void barrier(RankSignals sg, Signal* self_sg, int rank) {
  if (threadIdx.x < ngpus) {
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto* pc = &sg.signals[threadIdx.x]->peer_counter[val%2][blockIdx.x][rank];
    auto* sc = &self_sg->peer_counter[val%2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) { st_rel(pc, val); while(ld_acq(sc)!=val); }
    else { st_vol(pc, val); while(ld_vol(sc)!=val); }
  }
  __syncthreads();
}

// ============================================================
// Test 1: Barrier only (no data, measure pure sync cost)
// ============================================================
template <int ngpus>
__global__ void kernel_barrier_only(RankSignals sg, Signal* self_sg, int rank) {
  barrier<ngpus, false>(sg, self_sg, rank);
  barrier<ngpus, true>(sg, self_sg, rank);
}

// ============================================================
// Test 2: PCIe read from peer (no sync, pull model)
// ============================================================
__global__ void kernel_read_peer(const float4* __restrict__ src, float4* __restrict__ dst, int n) {
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    dst[i] = src[i];
}

// ============================================================
// Test 3: PCIe write to peer (no sync, push model)
// ============================================================
__global__ void kernel_write_peer(const float4* __restrict__ src, float4* __restrict__ dst, int n) {
  // Same kernel as read, but caller swaps src/dst pointers:
  // src = local buffer, dst = peer buffer (PCIe write)
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    dst[i] = src[i];
}

// ============================================================
// Test 4: Write with LL-style flag (4B data + 4B flag per 8B)
// Simulates Ring LL's per-chunk flagging
// ============================================================
struct LLPack {
  uint32_t data; uint32_t flag;
  __host__ __device__ volatile LLPack& operator=(const LLPack& o) volatile {
    data = o.data; flag = o.flag; return *this;
  }
};

__global__ void kernel_write_with_flag(const uint32_t* __restrict__ src,
                                        volatile LLPack* __restrict__ dst,
                                        int n_chunks, uint32_t flag_val) {
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n_chunks; i += gridDim.x*blockDim.x) {
    LLPack p;
    p.data = src[i];
    p.flag = flag_val;
    dst[i] = p;  // 8B atomic-ish write to peer
  }
}

// Receiver: spin on LL flags (local reads)
__global__ void kernel_recv_ll_flag(volatile LLPack* __restrict__ buf,
                                     uint32_t* __restrict__ result,
                                     int n_chunks, uint32_t expected_flag) {
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n_chunks; i += gridDim.x*blockDim.x) {
    // Spin until flag matches
    while (buf[i].flag != expected_flag) {}
    result[i] = buf[i].data;
  }
}

// ============================================================
// Test 5: Oneshot full (barrier + read + reduce + barrier)
// ============================================================
template <int ngpus>
__global__ void kernel_oneshot_full(RankSignals sg, Signal* self_sg,
                                     const float4* __restrict__* ptrs,
                                     float4* __restrict__ result,
                                     int rank, int n) {
  barrier<ngpus, false>(sg, self_sg, rank);
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
    float4 sum = ptrs[0][i];
    for (int r = 1; r < ngpus; r++) {
      float4 v = ptrs[r][i];
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    result[i] = sum;
  }
  barrier<ngpus, true>(sg, self_sg, rank);
}

// ============================================================
// Benchmark harness
// ============================================================
struct GPUState {
  float4* data;
  float4* peer_data;  // mapped peer buffer
  uint32_t* data_u32;
  LLPack* ll_recv_buf;
  uint32_t* ll_result;
  Signal* signal;
  cudaStream_t stream;
  cudaEvent_t start, stop;
};

struct BenchArgs {
  int rank, ngpus, iters;
  size_t bytes;
  GPUState* states;
  RankSignals sg;
  pthread_barrier_t* barrier;
  float times[6];  // median for each test
};

float run_test(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream,
               pthread_barrier_t* bar, int ngpus, int iters,
               auto launch) {
  // Warmup
  for (int w = 0; w < 20; w++) {
    pthread_barrier_wait(bar);
    launch();
    CHECK(cudaStreamSynchronize(stream));
  }
  pthread_barrier_wait(bar);

  float total = 0;
  float times[2048];
  for (int it = 0; it < iters; it++) {
    pthread_barrier_wait(bar);
    CHECK(cudaEventRecord(start, stream));
    launch();
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    times[it] = ms * 1000.0f;
    total += times[it];
  }
  // Sort, return median
  for (int i = 0; i < iters-1; i++)
    for (int j = i+1; j < iters; j++)
      if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
  return times[iters/2];
}

template <int ngpus>
void* bench_thread(void* arg) {
  BenchArgs* a = (BenchArgs*)arg;
  int rank = a->rank;
  int peer = (rank + 1) % a->ngpus;
  CHECK(cudaSetDevice(rank));

  auto& s = a->states[rank];
  int n4 = a->bytes / sizeof(float4);
  int n_chunks = a->bytes / sizeof(uint32_t);
  int nblocks = min((n4 + 255) / 256, kMaxBlocks);
  if (nblocks < 1) nblocks = 1;

  CHECK(cudaMemset(s.data, 1, a->bytes));
  CHECK(cudaMemset(s.signal, 0, sizeof(Signal)));
  if (s.ll_recv_buf) CHECK(cudaMemset(s.ll_recv_buf, 0, n_chunks * sizeof(LLPack)));
  pthread_barrier_wait(a->barrier);

  // Test 1: Barrier only
  a->times[0] = run_test(s.start, s.stop, s.stream, a->barrier, a->ngpus, a->iters,
    [&]() {
      kernel_barrier_only<ngpus><<<1, 512, 0, s.stream>>>(a->sg, s.signal, rank);
    });

  // Test 2: Read from peer (pull)
  a->times[1] = run_test(s.start, s.stop, s.stream, a->barrier, a->ngpus, a->iters,
    [&]() {
      kernel_read_peer<<<nblocks, 256, 0, s.stream>>>(
        a->states[peer].data, s.data, n4);
    });

  // Test 3: Write to peer (push)
  a->times[2] = run_test(s.start, s.stop, s.stream, a->barrier, a->ngpus, a->iters,
    [&]() {
      kernel_write_peer<<<nblocks, 256, 0, s.stream>>>(
        s.data, a->states[peer].data, n4);
    });

  // Test 4: Write with LL flag (Ring LL simulation)
  // Skip if ll_recv_buf not allocated
  if (s.ll_recv_buf) {
    static uint32_t flag_counter = 1;
    a->times[3] = run_test(s.start, s.stop, s.stream, a->barrier, a->ngpus, a->iters,
      [&]() {
        uint32_t f = ++flag_counter;
        // Sender writes to peer's LL buffer
        kernel_write_with_flag<<<nblocks, 256, 0, s.stream>>>(
          s.data_u32, (volatile LLPack*)a->states[peer].ll_recv_buf, n_chunks, f);
      });
  } else {
    a->times[3] = 0;
  }

  // Test 5: Oneshot full (barrier + read + reduce + barrier)
  {
    // Setup pointer array on device
    float4* h_ptrs[8];
    for (int r = 0; r < a->ngpus; r++) h_ptrs[r] = a->states[r].data;
    float4** d_ptrs;
    CHECK(cudaMalloc(&d_ptrs, 8 * sizeof(float4*)));
    CHECK(cudaMemcpy(d_ptrs, h_ptrs, a->ngpus * sizeof(float4*), cudaMemcpyHostToDevice));

    CHECK(cudaMemset(s.signal, 0, sizeof(Signal)));
    pthread_barrier_wait(a->barrier);

    a->times[4] = run_test(s.start, s.stop, s.stream, a->barrier, a->ngpus, a->iters,
      [&]() {
        kernel_oneshot_full<ngpus><<<nblocks, 256, 0, s.stream>>>(
          a->sg, s.signal, (const float4* __restrict__*)d_ptrs, s.data, rank, n4);
      });
    CHECK(cudaFree(d_ptrs));
  }

  // Test 6: Two-phase write (simulate Ring reduce-scatter + all-gather)
  a->times[5] = run_test(s.start, s.stop, s.stream, a->barrier, a->ngpus, a->iters,
    [&]() {
      // Phase 1: write first half to peer
      kernel_write_peer<<<nblocks, 256, 0, s.stream>>>(
        s.data, a->states[peer].data, n4/2);
      // Phase 2: write second half to peer
      kernel_write_peer<<<nblocks, 256, 0, s.stream>>>(
        s.data + n4/2, a->states[peer].data + n4/2, n4/2);
    });

  return nullptr;
}

int main(int argc, char** argv) {
  int ngpus = argc > 1 ? atoi(argv[1]) : 2;
  size_t bytes = argc > 2 ? atoi(argv[2]) : 4096;
  int iters = argc > 3 ? atoi(argv[3]) : 200;

  if (ngpus != 2 && ngpus != 4) {
    fprintf(stderr, "Only 2 or 4 GPUs\n"); return 1;
  }

  // Setup
  GPUState states[8];
  for (int i = 0; i < ngpus; i++) {
    CHECK(cudaSetDevice(i));
    for (int j = 0; j < ngpus; j++)
      if (i!=j) { int c; CHECK(cudaDeviceCanAccessPeer(&c,i,j)); if(c) CHECK(cudaDeviceEnablePeerAccess(j,0)); }
    CHECK(cudaMalloc(&states[i].data, bytes));
    states[i].data_u32 = (uint32_t*)states[i].data;
    CHECK(cudaMalloc(&states[i].ll_recv_buf, (bytes/4) * sizeof(LLPack)));
    CHECK(cudaMalloc(&states[i].ll_result, bytes));
    CHECK(cudaMalloc(&states[i].signal, sizeof(Signal)));
    CHECK(cudaMemset(states[i].signal, 0, sizeof(Signal)));
    CHECK(cudaStreamCreate(&states[i].stream));
    CHECK(cudaEventCreate(&states[i].start));
    CHECK(cudaEventCreate(&states[i].stop));
  }

  RankSignals sg;
  for (int i = 0; i < ngpus; i++) sg.signals[i] = states[i].signal;

  printf("AllReduce Breakdown Benchmark\n");
  printf("  GPUs: %d, Data: %zu bytes, Iters: %d\n\n", ngpus, bytes, iters);

  pthread_barrier_t bar;
  pthread_barrier_init(&bar, nullptr, ngpus);

  BenchArgs args[8];
  pthread_t threads[8];
  for (int r = 0; r < ngpus; r++) {
    args[r] = {r, ngpus, iters, bytes, states, sg, &bar, {}};
    if (ngpus == 2) pthread_create(&threads[r], nullptr, bench_thread<2>, &args[r]);
    else pthread_create(&threads[r], nullptr, bench_thread<4>, &args[r]);
  }
  for (int r = 0; r < ngpus; r++) pthread_join(threads[r], nullptr);
  pthread_barrier_destroy(&bar);

  // Report max across GPUs
  printf("%-30s %10s\n", "Test", "Median (µs)");
  printf("──────────────────────────────────────────\n");
  const char* names[] = {
    "1. Barrier only (no data)",
    "2. Read peer (pull, no sync)",
    "3. Write peer (push, no sync)",
    "4. Write + LL flag (Ring sim)",
    "5. Oneshot full (bar+read+bar)",
    "6. Two-phase write (Ring sim)",
  };
  for (int t = 0; t < 6; t++) {
    float mx = 0;
    for (int r = 0; r < ngpus; r++) if (args[r].times[t] > mx) mx = args[r].times[t];
    printf("%-30s %10.1f\n", names[t], mx);
  }

  printf("\nAnalysis:\n");
  float barrier = 0, read = 0, write = 0, ll_flag = 0, oneshot = 0, two_phase = 0;
  for (int r = 0; r < ngpus; r++) {
    if (args[r].times[0] > barrier) barrier = args[r].times[0];
    if (args[r].times[1] > read) read = args[r].times[1];
    if (args[r].times[2] > write) write = args[r].times[2];
    if (args[r].times[3] > ll_flag) ll_flag = args[r].times[3];
    if (args[r].times[4] > oneshot) oneshot = args[r].times[4];
    if (args[r].times[5] > two_phase) two_phase = args[r].times[5];
  }
  printf("  Sync overhead:    barrier=%.1fus\n", barrier);
  printf("  Data direction:   read=%.1fus vs write=%.1fus (diff=%.1fus)\n", read, write, write-read);
  printf("  LL flag overhead: ll_write=%.1fus vs plain_write=%.1fus (diff=%.1fus)\n", ll_flag, write, ll_flag-write);
  printf("  Phase overhead:   two_phase=%.1fus vs one_phase_write=%.1fus (diff=%.1fus)\n", two_phase, write, two_phase-write);
  printf("  Oneshot total:    %.1fus = barrier(%.1f) + read(%.1f) + reduce_overhead(%.1f)\n",
         oneshot, barrier, read, oneshot - barrier - read);

  // Cleanup
  for (int i = 0; i < ngpus; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(states[i].data)); CHECK(cudaFree(states[i].ll_recv_buf));
    CHECK(cudaFree(states[i].ll_result)); CHECK(cudaFree(states[i].signal));
  }
  return 0;
}
