/*
 * Standalone benchmark for vLLM-style custom all-reduce.
 * Uses nccl-tests style: one thread per GPU, pthread_barrier sync,
 * per-GPU CUDA Graph, per-GPU timing (report max across GPUs).
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_80 bench_oneshot.cu -o bench_oneshot -lpthread
 *
 * Run:
 *   ./bench_oneshot [num_gpus] [warmup] [iters] [size_bytes] [repeats] [direct_warmup] [algo]
 *   ./bench_oneshot 2 50 200
 *   ./bench_oneshot 4 100 500
 *   ./bench_oneshot 4 0 3 2147483648 1 1 twoshot
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
constexpr size_t kAllReduceSmallThreshold = 512 * 1024;
constexpr size_t kAllReduceLargeThreshold = 256 * 1024;

enum class Algo {
  Auto = 0,
  OneShot = 1,
  TwoShot = 2,
};

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

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];

  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);

  for (int idx = tid; idx < largest_part; idx += stride) {
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = (rank + i) % ngpus;
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

const char* algo_name(Algo algo) {
  switch (algo) {
    case Algo::Auto: return "auto";
    case Algo::OneShot: return "oneshot";
    case Algo::TwoShot: return "twoshot";
  }
  return "unknown";
}

Algo parse_algo(const char* s) {
  if (strcmp(s, "auto") == 0) return Algo::Auto;
  if (strcmp(s, "1stage") == 0 || strcmp(s, "oneshot") == 0) return Algo::OneShot;
  if (strcmp(s, "2stage") == 0 || strcmp(s, "twoshot") == 0) return Algo::TwoShot;
  fprintf(stderr, "Invalid algo '%s'. Valid values: auto, oneshot, twoshot\n", s);
  exit(1);
}

Algo resolve_algo(Algo requested, int ngpus, size_t bytes) {
  if (requested != Algo::Auto) return requested;
  if (ngpus == 2) return Algo::OneShot;
  size_t threshold = ngpus <= 4 ? kAllReduceSmallThreshold : kAllReduceLargeThreshold;
  return bytes < threshold ? Algo::OneShot : Algo::TwoShot;
}

template <int ngpus>
void launch_reduce(Algo algo, RankData* rd, RankSignals sg, Signal* signal,
                   half* data, int rank, int packed_size, int nblocks,
                   cudaStream_t stream) {
  if (algo == Algo::TwoShot) {
    cross_device_reduce_2stage<half, ngpus><<<nblocks, 512, 0, stream>>>(
        rd, sg, signal, data, rank, packed_size);
  } else {
    cross_device_reduce_1stage<half, ngpus><<<nblocks, 512, 0, stream>>>(
        rd, sg, signal, data, rank, packed_size);
  }
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
  int repeats, direct_warmup;
  Algo requested_algo;
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
  Algo algo = resolve_algo(a->requested_algo, a->ngpus, a->bytes);

  cudaStream_t stream = a->states[rank].stream;
  Signal* signal = a->states[rank].signal;
  RankData* rd = a->states[rank].rank_data;
  CHECK_CUDA(cudaMemset(a->states[rank].data, 1, a->bytes));
  CHECK_CUDA(cudaMemset(signal, 0, sizeof(Signal)));
  pthread_barrier_wait(a->barrier);  // all GPUs ready

  // Warmup (direct launch, kernel internal barrier syncs GPUs)
  for (int w = 0; w < a->direct_warmup; w++) {
    launch_reduce<ngpus>(algo, rd, a->sg, signal, a->states[rank].data, rank, packed_size, nblocks, stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  pthread_barrier_wait(a->barrier);

  // Capture graph (20 repeats)
  cudaGraph_t graph;
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int rep = 0; rep < a->repeats; rep++)
    launch_reduce<ngpus>(algo, rd, a->sg, signal, a->states[rank].data, rank, packed_size, nblocks, stream);
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
    a->my_time_us[it] = ms * 1000.0f / a->repeats;
  }

  cudaGraphExecDestroy(a->states[rank].graph_exec);
  return nullptr;
}

void bench_size(GPUState* states, int ngpus, size_t bytes, int warmup, int iters,
                int repeats, int direct_warmup, Algo requested_algo) {
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, nullptr, ngpus);

  RankSignals sg;
  for (int i = 0; i < ngpus; i++) sg.signals[i] = states[i].signal;

  BenchArgs args[8];
  pthread_t threads[8];
  for (int r = 0; r < ngpus; r++) {
    args[r] = {r, ngpus, warmup, iters, repeats, direct_warmup, requested_algo, bytes, states, sg, &barrier, {}};
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
    CHECK_CUDA(cudaMalloc(&states[i].signal, sizeof(Signal) + max_bytes));
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
  size_t single_size = argc > 4 ? strtoull(argv[4], nullptr, 0) : 0;
  int repeats = argc > 5 ? atoi(argv[5]) : 20;
  int direct_warmup = argc > 6 ? atoi(argv[6]) : 5;
  Algo requested_algo = argc > 7 ? parse_algo(argv[7]) : Algo::OneShot;

  if (ngpus != 2 && ngpus != 4 && ngpus != 8) {
    fprintf(stderr, "Only 2, 4, 8 GPUs supported\n"); return 1;
  }

  size_t default_sizes[] = {1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
  size_t single_sizes[] = {single_size};
  size_t* sizes = single_size ? single_sizes : default_sizes;
  int nsizes = single_size ? 1 : (int)(sizeof(default_sizes) / sizeof(default_sizes[0]));

  GPUState states[8];
  setup_gpus(states, ngpus, sizes[nsizes-1]);

  printf("Custom All-Reduce Benchmark [nccl-tests style: pthread + CUDA Graph + per-GPU timing]\n");
  printf("  GPUs: %d, Algo: %s, Direct warmup: %d, Graph warmup: %d, Iters: %d, Repeats/graph: %d\n\n",
         ngpus, algo_name(requested_algo), direct_warmup, warmup, iters, repeats);
  printf("%8s    %12s %12s %12s\n", "Size", "Median (µs)", "Mean (µs)", "P99 (µs)");
  printf("------------------------------------------------------\n");

  for (int s = 0; s < nsizes; s++)
    bench_size(states, ngpus, sizes[s], warmup, iters, repeats, direct_warmup, requested_algo);

  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(states[i].data)); CHECK_CUDA(cudaFree(states[i].signal));
    CHECK_CUDA(cudaFree(states[i].rank_data)); CHECK_CUDA(cudaStreamDestroy(states[i].stream));
    CHECK_CUDA(cudaEventDestroy(states[i].start)); CHECK_CUDA(cudaEventDestroy(states[i].stop));
  }
  return 0;
}
