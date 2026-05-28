/*
 * Standalone benchmark for vLLM-style 1-stage (oneshot) all-reduce.
 * Extracted from vllm/sglang custom_all_reduce.cuh (Apache 2.0).
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_80 bench_oneshot.cu -o bench_oneshot
 *   # For PPU: replace nvcc with PPU compiler, adjust -arch
 *
 * Run:
 *   # Single process, 2 GPUs (uses cudaEnablePeerAccess, no IPC)
 *   ./bench_oneshot [num_gpus] [warmup] [iters]
 *   ./bench_oneshot 2 50 200
 */

#include <thread>
#include <atomic>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t e = (call);                                              \
    if (e != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(e));                                     \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

// ============================================================
// Kernel types (from custom_all_reduce.cuh)
// ============================================================
using FlagType = uint32_t;

constexpr int kMaxBlocks = 36;

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
};

struct __align__(16) RankData {
  const void* __restrict__ ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(half val) { return __half2float(val); }
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }

template <typename T> DINLINE T downcast_s(float val);
template <> DINLINE half downcast_s(float val) { return __float2half(val); }
template <> DINLINE nv_bfloat16 downcast_s(float val) { return __float2bfloat16(val); }

DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) { a = __hadd(a, b); return a; }
DINLINE float& assign_add(float& a, float b) { return a += b; }

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
  #pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) return val;
  array_t<float, N> out;
  #pragma unroll
  for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
  return out;
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) return val;
  O out;
  #pragma unroll
  for (int i = 0; i < O::size; i++) out.data[i] = downcast_s<typename O::type>(val.data[i]);
  return out;
}

// ============================================================
// Barriers (system-scope release/acquire)
// ============================================================
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  if (threadIdx.x < ngpus) {
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) {
      st_flag_release(peer_counter_ptr, val);
      while (ld_flag_acquire(self_counter_ptr) != val);
    } else {
      st_flag_volatile(peer_counter_ptr, val);
      while (ld_flag_volatile(self_counter_ptr) != val);
    }
  }
  if constexpr (is_start || need_fence) __syncthreads();
}

// ============================================================
// 1-stage (oneshot) all-reduce kernel
// ============================================================
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
  #pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);
}

// ============================================================
// Host benchmark code
// ============================================================
struct GPUState {
  int device;
  half* data;       // input/output buffer
  Signal* signal;   // sync signals
  RankData* rank_data;  // device-side rank data
  cudaStream_t stream;
  cudaEvent_t start, stop;
};

void setup_gpus(GPUState* states, int ngpus, size_t max_bytes) {
  // Enable peer access
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int j = 0; j < ngpus; j++) {
      if (i != j) {
        int can;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&can, i, j));
        if (can) CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
  }

  // Allocate buffers and signals
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&states[i].data, max_bytes));
    CHECK_CUDA(cudaMalloc(&states[i].signal, sizeof(Signal)));
    CHECK_CUDA(cudaMemset(states[i].signal, 0, sizeof(Signal)));
    CHECK_CUDA(cudaMalloc(&states[i].rank_data, sizeof(RankData)));
    CHECK_CUDA(cudaStreamCreate(&states[i].stream));
    CHECK_CUDA(cudaEventCreate(&states[i].start));
    CHECK_CUDA(cudaEventCreate(&states[i].stop));
  }

  // Setup rank data (each GPU gets pointers to all GPUs' data)
  for (int i = 0; i < ngpus; i++) {
    RankData h_rd;
    for (int j = 0; j < ngpus; j++) h_rd.ptrs[j] = states[j].data;
    for (int j = ngpus; j < 8; j++) h_rd.ptrs[j] = nullptr;
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpy(states[i].rank_data, &h_rd, sizeof(RankData),
                           cudaMemcpyHostToDevice));
  }
}

typedef void (*launch_fn)(GPUState*, int, RankSignals, int, int);

template <typename T, int ngpus>
void launch_1stage(GPUState* states, int rank, RankSignals sg,
                   int packed_size, int nblocks) {
  cross_device_reduce_1stage<T, ngpus><<<nblocks, 512, 0, states[rank].stream>>>(
      states[rank].rank_data, sg, states[rank].signal,
      (T*)states[rank].data, rank, packed_size);
}

void bench_size(GPUState* states, int ngpus, size_t bytes,
                int warmup, int iters) {
  int packed_size = bytes / sizeof(half) / (16 / sizeof(half));
  int nblocks = std::min((packed_size + 511) / 512, kMaxBlocks);
  if (nblocks < 1) nblocks = 1;

  RankSignals sg;
  for (int i = 0; i < ngpus; i++) sg.signals[i] = states[i].signal;

  // Fill with test data
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemset(states[i].data, 1, bytes));
  }

  // Reset signal counters
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemset(states[i].signal, 0, sizeof(Signal)));
  }

  // Lambda to launch kernel on all GPUs
  auto launch_all = [&]() {
    for (int r = 0; r < ngpus; r++) {
      CHECK_CUDA(cudaSetDevice(r));
      if (ngpus == 2)
        launch_1stage<half, 2>(states, r, sg, packed_size, nblocks);
      else if (ngpus == 4)
        launch_1stage<half, 4>(states, r, sg, packed_size, nblocks);
      else if (ngpus == 8)
        launch_1stage<half, 8>(states, r, sg, packed_size, nblocks);
    }
  };

  // Warmup (direct launch)
  for (int w = 0; w < 5; w++) {
    launch_all();
    for (int r = 0; r < ngpus; r++) {
      CHECK_CUDA(cudaSetDevice(r));
      CHECK_CUDA(cudaStreamSynchronize(states[r].stream));
    }
  }

  // Capture per-GPU CUDA Graphs with N repeats per graph
  // This amortizes graph launch overhead across N allreduce calls
  const int REPEATS_PER_GRAPH = 20;
  cudaGraph_t graphs[8];
  cudaGraphExec_t graph_execs[8];
  for (int r = 0; r < ngpus; r++) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamBeginCapture(states[r].stream, cudaStreamCaptureModeThreadLocal));
    for (int rep = 0; rep < REPEATS_PER_GRAPH; rep++) {
      if (ngpus == 2) launch_1stage<half, 2>(states, r, sg, packed_size, nblocks);
      else if (ngpus == 4) launch_1stage<half, 4>(states, r, sg, packed_size, nblocks);
      else if (ngpus == 8) launch_1stage<half, 8>(states, r, sg, packed_size, nblocks);
    }
    CHECK_CUDA(cudaStreamEndCapture(states[r].stream, &graphs[r]));
    CHECK_CUDA(cudaGraphInstantiate(&graph_execs[r], graphs[r], nullptr, nullptr, 0));
  }

  // Thread-parallel graph launch: all GPUs launch simultaneously
  auto parallel_launch = [&](cudaGraphExec_t* execs) {
    std::atomic<int> ready{0};
    std::thread threads[8];
    for (int r = 0; r < ngpus; r++) {
      threads[r] = std::thread([&, r]() {
        CHECK_CUDA(cudaSetDevice(r));
        ready.fetch_add(1);
        while (ready.load() < ngpus) {}  // spin until all ready
        CHECK_CUDA(cudaGraphLaunch(execs[r], states[r].stream));
      });
    }
    for (int r = 0; r < ngpus; r++) threads[r].join();
  };

  // Warmup with parallel graph replay
  for (int w = 0; w < warmup; w++) {
    parallel_launch(graph_execs);
    for (int r = 0; r < ngpus; r++) {
      CHECK_CUDA(cudaSetDevice(r));
      CHECK_CUDA(cudaStreamSynchronize(states[r].stream));
    }
  }

  // Benchmark
  float total_us = 0;
  float times[4096];
  for (int it = 0; it < iters; it++) {
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(states[0].start, states[0].stream));

    parallel_launch(graph_execs);

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(states[0].stop, states[0].stream));
    for (int r = 0; r < ngpus; r++) {
      CHECK_CUDA(cudaSetDevice(r));
      CHECK_CUDA(cudaStreamSynchronize(states[r].stream));
    }

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, states[0].start, states[0].stop));
    ms /= REPEATS_PER_GRAPH;  // per-allreduce time
    times[it] = ms * 1000.0f;
    total_us += times[it];
  }

  for (int r = 0; r < ngpus; r++) {
    cudaGraphExecDestroy(graph_execs[r]);
    cudaGraphDestroy(graphs[r]);
  }

  // Sort and compute median
  for (int i = 0; i < iters - 1; i++)
    for (int j = i + 1; j < iters; j++)
      if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }

  float median = times[iters / 2];
  float mean = total_us / iters;
  float p99 = times[(int)(iters * 0.99)];

  const char* unit = "B";
  float display_size = bytes;
  if (bytes >= 1024 * 1024) { display_size = bytes / (1024.0 * 1024.0); unit = "MB"; }
  else if (bytes >= 1024) { display_size = bytes / 1024.0; unit = "KB"; }

  printf("%8.0f %-2s %12.1f %12.1f %12.1f\n", display_size, unit, median, mean, p99);
}

int main(int argc, char** argv) {
  int ngpus = argc > 1 ? atoi(argv[1]) : 2;
  int warmup = argc > 2 ? atoi(argv[2]) : 50;
  int iters = argc > 3 ? atoi(argv[3]) : 200;

  printf("Oneshot (1-stage) All-Reduce Benchmark\n");
  printf("  GPUs: %d, Warmup: %d, Iters: %d\n\n", ngpus, warmup, iters);

  if (ngpus != 2 && ngpus != 4 && ngpus != 8) {
    fprintf(stderr, "Only 2, 4, 8 GPUs supported\n");
    return 1;
  }

  size_t sizes[] = {1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);
  size_t max_bytes = sizes[nsizes - 1];

  GPUState states[8];
  for (int i = 0; i < ngpus; i++) states[i].device = i;
  setup_gpus(states, ngpus, max_bytes);

  printf("%8s    %12s %12s %12s\n", "Size", "Median (µs)", "Mean (µs)", "P99 (µs)");
  printf("------------------------------------------------------\n");

  for (int s = 0; s < nsizes; s++) {
    bench_size(states, ngpus, sizes[s], warmup, iters);
  }

  // Cleanup
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(states[i].data));
    CHECK_CUDA(cudaFree(states[i].signal));
    CHECK_CUDA(cudaFree(states[i].rank_data));
    CHECK_CUDA(cudaStreamDestroy(states[i].stream));
    CHECK_CUDA(cudaEventDestroy(states[i].start));
    CHECK_CUDA(cudaEventDestroy(states[i].stop));
  }

  return 0;
}
