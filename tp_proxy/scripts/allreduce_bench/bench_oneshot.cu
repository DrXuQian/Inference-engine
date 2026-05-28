/*
 * Standalone benchmark for vLLM-style 1-stage (oneshot) all-reduce.
 * Uses nccl-tests style: one thread per GPU, pthread_barrier sync,
 * per-GPU CUDA Graph, per-GPU timing (report max across GPUs).
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_80 bench_oneshot.cu -o bench_oneshot -lpthread
 *
 * Run:
 *   ./bench_oneshot [num_gpus] [warmup] [iters]
 *   ./bench_oneshot 2 50 200
 *   ./bench_oneshot 4 100 500
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pthread.h>
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
// Kernel types
// ============================================================
using FlagType = uint32_t;
constexpr int kMaxBlocks = 36;

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
};
struct __align__(16) RankData { const void* __restrict__ ptrs[8]; };
struct __align__(16) RankSignals { Signal* signals[8]; };
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz]; using type = T; static constexpr int size = sz;
};
template <typename T> struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__
DINLINE float upcast_s(half v) { return __half2float(v); }
DINLINE float upcast_s(nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> DINLINE T downcast_s(float v);
template <> DINLINE half downcast_s(float v) { return __float2half(v); }
template <> DINLINE nv_bfloat16 downcast_s(float v) { return __float2bfloat16(v); }
DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) { a = __hadd(a, b); return a; }
DINLINE float& assign_add(float& a, float b) { return a += b; }
template <typename T, int N>
DINLINE array_t<T,N>& packed_assign_add(array_t<T,N>& a, array_t<T,N> b) {
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]); return a;
}
template <typename T, int N>
DINLINE array_t<float,N> upcast(array_t<T,N> v) {
  if constexpr (std::is_same<T,float>::value) return v;
  array_t<float,N> o; for (int i = 0; i < N; i++) o.data[i] = upcast_s(v.data[i]); return o;
}
template <typename O>
DINLINE O downcast(array_t<float,O::size> v) {
  if constexpr (std::is_same<typename O::type,float>::value) return v;
  O o; for (int i = 0; i < O::size; i++) o.data[i] = downcast_s<typename O::type>(v.data[i]); return o;
}

// Barriers
static DINLINE void st_flag_release(FlagType* a, FlagType f) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(f), "l"(a));
}
static DINLINE FlagType ld_flag_acquire(FlagType* a) {
  FlagType f; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(f) : "l"(a)); return f;
}
static DINLINE void st_flag_volatile(FlagType* a, FlagType f) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(f), "l"(a));
}
static DINLINE FlagType ld_flag_volatile(FlagType* a) {
  FlagType f; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(f) : "l"(a)); return f;
}
template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  if (threadIdx.x < ngpus) {
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto* pc = &sg.signals[threadIdx.x]->peer_counter[val%2][blockIdx.x][rank];
    auto* sc = &self_sg->peer_counter[val%2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) { st_flag_release(pc, val); while (ld_flag_acquire(sc) != val); }
    else { st_flag_volatile(pc, val); while (ld_flag_volatile(sc) != val); }
  }
  if constexpr (is_start || need_fence) __syncthreads();
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P; using A = typename packed_t<T>::A;
  auto dp = *_dp;
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x*blockDim.x+threadIdx.x; idx < size; idx += gridDim.x*blockDim.x)
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);
}

// ============================================================
// Shared state across threads (nccl-tests style)
// ============================================================
struct GPUState {
  half* data;
  Signal* signal;
  RankData* rank_data;
  cudaStream_t stream;
  cudaEvent_t start, stop;
  cudaGraphExec_t graph_exec;
};

struct BenchArgs {
  int rank, ngpus, warmup, iters;
  size_t bytes;
  GPUState* states;
  RankSignals sg;
  pthread_barrier_t* barrier;
  float my_time_us[4096];  // per-iter timing for this GPU
};

template <int ngpus>
void* bench_thread(void* arg) {
  BenchArgs* a = (BenchArgs*)arg;
  int rank = a->rank;
  CHECK_CUDA(cudaSetDevice(rank));

  int packed_size = a->bytes / sizeof(half) / (16 / sizeof(half));
  int nblocks = std::min((packed_size + 511) / 512, kMaxBlocks);
  if (nblocks < 1) nblocks = 1;

  cudaStream_t stream = a->states[rank].stream;
  Signal* signal = a->states[rank].signal;
  RankData* rd = a->states[rank].rank_data;
  const int REPEATS = 20;

  CHECK_CUDA(cudaMemset(a->states[rank].data, 1, a->bytes));
  CHECK_CUDA(cudaMemset(signal, 0, sizeof(Signal)));
  pthread_barrier_wait(a->barrier);  // all GPUs ready

  // Warmup (direct launch, kernel internal barrier syncs GPUs)
  for (int w = 0; w < 5; w++) {
    cross_device_reduce_1stage<half, ngpus><<<nblocks, 512, 0, stream>>>(
        rd, a->sg, signal, a->states[rank].data, rank, packed_size);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  pthread_barrier_wait(a->barrier);

  // Capture graph (20 repeats)
  cudaGraph_t graph;
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int rep = 0; rep < REPEATS; rep++)
    cross_device_reduce_1stage<half, ngpus><<<nblocks, 512, 0, stream>>>(
        rd, a->sg, signal, a->states[rank].data, rank, packed_size);
  CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
  CHECK_CUDA(cudaGraphInstantiate(&a->states[rank].graph_exec, graph, nullptr, nullptr, 0));
  cudaGraphDestroy(graph);
  pthread_barrier_wait(a->barrier);  // all graphs captured

  // Graph warmup (kernel barrier syncs GPUs, pthread barrier between rounds)
  for (int w = 0; w < a->warmup; w++) {
    pthread_barrier_wait(a->barrier);
    CHECK_CUDA(cudaGraphLaunch(a->states[rank].graph_exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  pthread_barrier_wait(a->barrier);

  // Benchmark: each GPU times itself, pthread barrier ensures simultaneous launch
  cudaEvent_t ev_start = a->states[rank].start;
  cudaEvent_t ev_stop = a->states[rank].stop;
  for (int it = 0; it < a->iters; it++) {
    pthread_barrier_wait(a->barrier);  // all GPUs launch together
    CHECK_CUDA(cudaEventRecord(ev_start, stream));
    CHECK_CUDA(cudaGraphLaunch(a->states[rank].graph_exec, stream));
    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    a->my_time_us[it] = ms * 1000.0f / REPEATS;
  }

  cudaGraphExecDestroy(a->states[rank].graph_exec);
  return nullptr;
}

void bench_size(GPUState* states, int ngpus, size_t bytes, int warmup, int iters) {
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, nullptr, ngpus);

  RankSignals sg;
  for (int i = 0; i < ngpus; i++) sg.signals[i] = states[i].signal;

  BenchArgs args[8];
  pthread_t threads[8];
  for (int r = 0; r < ngpus; r++) {
    args[r] = {r, ngpus, warmup, iters, bytes, states, sg, &barrier, {}};
    if (ngpus == 2) pthread_create(&threads[r], nullptr, bench_thread<2>, &args[r]);
    else if (ngpus == 4) pthread_create(&threads[r], nullptr, bench_thread<4>, &args[r]);
    else if (ngpus == 8) pthread_create(&threads[r], nullptr, bench_thread<8>, &args[r]);
  }
  for (int r = 0; r < ngpus; r++) pthread_join(threads[r], nullptr);
  pthread_barrier_destroy(&barrier);

  // Take MAX time across GPUs for each iteration (like nccl-tests)
  float times[4096];
  float total_us = 0;
  for (int it = 0; it < iters; it++) {
    float max_t = 0;
    for (int r = 0; r < ngpus; r++)
      if (args[r].my_time_us[it] > max_t) max_t = args[r].my_time_us[it];
    times[it] = max_t;
    total_us += max_t;
  }

  // Sort
  for (int i = 0; i < iters-1; i++)
    for (int j = i+1; j < iters; j++)
      if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }

  float median = times[iters/2];
  float mean = total_us / iters;
  float p99 = times[(int)(iters * 0.99)];

  const char* unit = "B"; float ds = bytes;
  if (bytes >= 1024*1024) { ds = bytes/(1024.0*1024.0); unit = "MB"; }
  else if (bytes >= 1024) { ds = bytes/1024.0; unit = "KB"; }
  printf("%8.0f %-2s %12.1f %12.1f %12.1f\n", ds, unit, median, mean, p99);
}

void setup_gpus(GPUState* states, int ngpus, size_t max_bytes) {
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int j = 0; j < ngpus; j++)
      if (i != j) { int c; CHECK_CUDA(cudaDeviceCanAccessPeer(&c, i, j)); if (c) CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0)); }
  }
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
  for (int i = 0; i < ngpus; i++) {
    RankData h_rd;
    for (int j = 0; j < ngpus; j++) h_rd.ptrs[j] = states[j].data;
    for (int j = ngpus; j < 8; j++) h_rd.ptrs[j] = nullptr;
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpy(states[i].rank_data, &h_rd, sizeof(RankData), cudaMemcpyHostToDevice));
  }
}

int main(int argc, char** argv) {
  int ngpus = argc > 1 ? atoi(argv[1]) : 2;
  int warmup = argc > 2 ? atoi(argv[2]) : 50;
  int iters = argc > 3 ? atoi(argv[3]) : 200;

  if (ngpus != 2 && ngpus != 4 && ngpus != 8) {
    fprintf(stderr, "Only 2, 4, 8 GPUs supported\n"); return 1;
  }

  size_t sizes[] = {1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  GPUState states[8];
  setup_gpus(states, ngpus, sizes[nsizes-1]);

  printf("Oneshot All-Reduce Benchmark [nccl-tests style: pthread + CUDA Graph + per-GPU timing]\n");
  printf("  GPUs: %d, Warmup: %d, Iters: %d, Repeats/graph: 20\n\n", ngpus, warmup, iters);
  printf("%8s    %12s %12s %12s\n", "Size", "Median (µs)", "Mean (µs)", "P99 (µs)");
  printf("------------------------------------------------------\n");

  for (int s = 0; s < nsizes; s++) bench_size(states, ngpus, sizes[s], warmup, iters);

  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(states[i].data)); CHECK_CUDA(cudaFree(states[i].signal));
    CHECK_CUDA(cudaFree(states[i].rank_data)); CHECK_CUDA(cudaStreamDestroy(states[i].stream));
    CHECK_CUDA(cudaEventDestroy(states[i].start)); CHECK_CUDA(cudaEventDestroy(states[i].stop));
  }
  return 0;
}
