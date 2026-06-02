/*
 * Standalone benchmark for vLLM 0.19.1 custom all-reduce kernels.
 *
 * This file extracts the dependency-free CUDA kernel logic from:
 *   vllm-project/vllm@v0.19.1 csrc/custom_all_reduce.cuh
 *
 * Differences from vLLM's Python integration:
 *   - no torch/vLLM/NCCL dependency;
 *   - input and output are always separate buffers, matching vLLM's
 *     out-of-place custom allreduce API;
 *   - 4-GPU PCIe is enabled in this standalone benchmark. vLLM's Python
 *     policy disables custom AR for >2 PCIe-only GPUs, but the CUDA kernels
 *     themselves support 2/4/6/8 ranks.
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_120 bench_vllm_allreduce.cu -o bench_vllm_allreduce -lpthread
 *
 * Run:
 *   ./bench_vllm_allreduce [ngpus] [warmup] [iters] [size_bytes] [repeats] [direct_warmup] [algo] [verify] [label|--label label]
 *   ./bench_vllm_allreduce 2 20 50 6144 20 0 auto 1
 *   ./bench_vllm_allreduce 4 20 50 6144 20 0 oneshot 1 --label same-numa
 *
 * Algos:
 *   auto    - vLLM threshold choice, with 4-GPU PCIe allowed here.
 *   oneshot - vLLM cross_device_reduce_1stage.
 *   twoshot - vLLM cross_device_reduce_2stage.
 *   push    - safe experimental push path: input -> peer scratch -> output.
 */

#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t e = (call);                                              \
    if (e != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(e));                                     \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

constexpr int kMaxBlocks = 36;
constexpr size_t kVllmSmallThreshold = 512 * 1024;
constexpr size_t kVllmLargeThreshold = 256 * 1024;
constexpr size_t kPushMaxBytes = 8 * 1024 * 1024;

using FlagType = uint32_t;

enum class Algo {
  Auto = 0,
  OneShot = 1,
  TwoShot = 2,
  Push = 3,
};

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];
  alignas(128) FlagType end[kMaxBlocks][8];
  alignas(128) FlagType _flag[kMaxBlocks];
  alignas(128) FlagType push_epoch[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* ptrs[8];
  void* push_buffers[8];
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

template <typename T>
DINLINE T downcast_s(float val);

template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}

DINLINE float& assign_add(float& a, float b) { return a += b; }

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
#endif
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T>
DINLINE bool is_pos_zero(T v) {
  return false;
}

template <>
DINLINE bool is_pos_zero(half v) {
  return *reinterpret_cast<uint16_t*>(&v) == 0x0000u;
}

template <>
DINLINE bool is_pos_zero(float v) {
  return *reinterpret_cast<uint32_t*>(&v) == 0x00000000u;
}

template <typename T>
DINLINE void clear_pos_zero(T& v) {}

template <>
DINLINE void clear_pos_zero(half& v) {
  uint16_t* bits = reinterpret_cast<uint16_t*>(&v);
  if (*bits == 0x0000u) *bits = 0x8000u;
}

template <>
DINLINE void clear_pos_zero(float& v) {
  uint32_t* bits = reinterpret_cast<uint32_t*>(&v);
  if (*bits == 0x00000000u) *bits = 0x80000000u;
}

template <typename P>
DINLINE void clear_pos_zero_packed(P& v) {
#pragma unroll
  for (int i = 0; i < P::size; i++) clear_pos_zero(v.data[i]);
}

template <typename P>
DINLINE bool has_pos_zero_packed(const P& v) {
  bool found = false;
#pragma unroll
  for (int i = 0; i < P::size; i++) found |= is_pos_zero(v.data[i]);
  return found;
}

template <typename P>
DINLINE P make_pos_zero_packed() {
  P v;
#pragma unroll
  for (int i = 0; i < P::size; i++) v.data[i] = typename P::type{};
  return v;
}

template <typename P>
DINLINE void ld_global_volatile_16B(P& x, const P* addr) {
  static_assert(alignof(P) == 16 && sizeof(P) == 16);
  uint4 val;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<P*>(&val);
}

template <typename P>
DINLINE void st_global_volatile_16B(const P& x, P* addr) {
  static_assert(alignof(P) == 16 && sizeof(P) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::
                   "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

template <typename P, int ngpus>
DINLINE P reduce_storage(P (&storage)[ngpus]) {
  using A = typename packed_t<typename P::type>::A;
  A tmp = upcast(storage[0]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(storage[i]));
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result,
    int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    reinterpret_cast<P*>(result)[idx] =
        packed_reduce<P, ngpus, A>(reinterpret_cast<const P**>(&dp.ptrs[0]),
                                   idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return reinterpret_cast<P*>(reinterpret_cast<Signal*>(sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result,
    int rank, int size) {
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
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = reinterpret_cast<const P*>(_dp->ptrs[target]);
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        reinterpret_cast<P*>(result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_push_safe(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    const T* __restrict__ input, T* __restrict__ output, int rank, int size,
    size_t buffer_bytes) {
  using P = typename packed_t<T>::P;
  auto dp = *_dp;
  // Synchronize at the beginning of every graph node so a fast rank cannot
  // reuse a scratch epoch before slower ranks have cleared it in the previous
  // node. This keeps graph repeats safe without adding a completion barrier to
  // the measured push path.
  barrier_at_start<ngpus>(sg, self_sg, rank);
  uint32_t epoch = self_sg->push_epoch[blockIdx.x] & 1u;
  size_t epoch_offset = epoch * ngpus * buffer_bytes;

  P* push_bufs[ngpus];
  P* poll_bufs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    push_bufs[i] = reinterpret_cast<P*>(
        reinterpret_cast<char*>(dp.push_buffers[i]) + epoch_offset +
        rank * buffer_bytes);
    poll_bufs[i] = reinterpret_cast<P*>(
        reinterpret_cast<char*>(dp.push_buffers[rank]) + epoch_offset +
        i * buffer_bytes);
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    P vec = reinterpret_cast<const P*>(input)[idx];
    clear_pos_zero_packed(vec);
#pragma unroll
    for (int i = 0; i < ngpus; i++) st_global_volatile_16B(vec, push_bufs[i] + idx);
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    P storage[ngpus];
    while (true) {
      bool has_pos_zero = false;
#pragma unroll
      for (int i = 0; i < ngpus; i++) {
        ld_global_volatile_16B(storage[i], poll_bufs[i] + idx);
        has_pos_zero |= has_pos_zero_packed(storage[i]);
      }
      if (!has_pos_zero) break;
    }
    reinterpret_cast<P*>(output)[idx] = reduce_storage<P, ngpus>(storage);

    P zeros = make_pos_zero_packed<P>();
#pragma unroll
    for (int i = 0; i < ngpus; i++) st_global_volatile_16B(zeros, poll_bufs[i] + idx);
  }

  __syncthreads();
  if (threadIdx.x == 0) self_sg->push_epoch[blockIdx.x] = (epoch + 1u) & 1u;
}

__global__ void fill_half_kernel(half* data, int nelems, float value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  half v = __float2half(value);
  for (int i = tid; i < nelems; i += stride) data[i] = v;
}

const char* algo_name(Algo algo) {
  switch (algo) {
    case Algo::Auto: return "auto";
    case Algo::OneShot: return "oneshot";
    case Algo::TwoShot: return "twoshot";
    case Algo::Push: return "push";
  }
  return "unknown";
}

void print_usage(const char* prog) {
  printf("vLLM custom all-reduce standalone benchmark\n");
  printf("\n");
  printf("Usage:\n");
  printf("  %s [ngpus] [warmup] [iters] [size_bytes] [repeats] [direct_warmup] [algo] [verify] [label]\n", prog);
  printf("  %s --help\n", prog);
  printf("\n");
  printf("Positional arguments:\n");
  printf("  ngpus          Number of local GPUs/ranks to use. Supported: 2, 4, 6, 8.\n");
  printf("                 Default: 2. Usually combine with CUDA_VISIBLE_DEVICES.\n");
  printf("\n");
  printf("  warmup         Number of CUDA Graph warmup launches before timing.\n");
  printf("                 Default: 20.\n");
  printf("\n");
  printf("  iters          Number of timed CUDA Graph launches. Supported: 1..4096.\n");
  printf("                 Reported median/mean/p99 are computed from these iterations.\n");
  printf("                 Default: 50.\n");
  printf("\n");
  printf("  size_bytes     Message size per rank, in bytes. Must be a positive multiple\n");
  printf("                 of 16 because kernels operate on 16-byte packed values.\n");
  printf("                 Default: 0, which runs the built-in sweep:\n");
  printf("                 1024, 4096, 6144, 8192, 16384, 32768, 65536,\n");
  printf("                 131072, 262144, 524288, 1048576.\n");
  printf("\n");
  printf("  repeats        Number of all-reduce launches captured inside one CUDA Graph.\n");
  printf("                 The measured graph time is divided by repeats, so larger\n");
  printf("                 values reduce timing noise for tiny messages. Default: 20.\n");
  printf("\n");
  printf("  direct_warmup  Number of direct, non-graph kernel launches before graph\n");
  printf("                 capture. Useful for warming P2P/TLB/cache state. Default: 0.\n");
  printf("\n");
  printf("  algo           Algorithm to run. Default: auto.\n");
  printf("                   auto    - vLLM policy: TP=2 uses oneshot; <=4 GPUs and\n");
  printf("                             size < 512KB uses oneshot; <=8 GPUs and\n");
  printf("                             size < 256KB uses oneshot; otherwise twoshot.\n");
  printf("                   oneshot - vLLM cross_device_reduce_1stage pull path.\n");
  printf("                             Aliases: 1stage.\n");
  printf("                   twoshot - vLLM cross_device_reduce_2stage path.\n");
  printf("                             Aliases: 2stage.\n");
  printf("                   push    - local safe experimental push path using scratch\n");
  printf("                             buffers; max size is 8MB in this benchmark.\n");
  printf("                             Alias: pushshot.\n");
  printf("\n");
  printf("  verify         1 to verify output before/after timing, 0 to skip verify.\n");
  printf("                 Default: 1. Expected value is sum(rank+1) across ranks.\n");
  printf("\n");
  printf("  label          Optional label printed as the first output column.\n");
  printf("                 Accepted forms: bare label, --label LABEL, --label=LABEL,\n");
  printf("                 label=LABEL. The misspelling --lable/lable= is also accepted.\n");
  printf("\n");
  printf("Output columns:\n");
  printf("  Label          User label, or '-' when unset.\n");
  printf("  Size           Message size, shown in B/KB/MB.\n");
  printf("  Algo           Resolved algorithm after auto selection.\n");
  printf("  Median/Mean/P99(us)\n");
  printf("                 Per all-reduce latency in microseconds. For each iteration,\n");
  printf("                 the benchmark takes the max time across ranks, then reports\n");
  printf("                 median/mean/p99 over iters.\n");
  printf("\n");
  printf("Examples:\n");
  printf("  CUDA_VISIBLE_DEVICES=0,1 %s 2 20 50 6144 100 0 oneshot 1 --label tp2-6kb\n", prog);
  printf("  CUDA_VISIBLE_DEVICES=0,1,2,3 %s 4 0 10 6144 100 0 twoshot 1 --label tp4-6kb\n", prog);
  printf("  CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile -o /tmp/ar %s 4 0 10 6144 100 0 push 0\n", prog);
}

Algo parse_algo(const char* s) {
  if (strcmp(s, "auto") == 0) return Algo::Auto;
  if (strcmp(s, "1stage") == 0 || strcmp(s, "oneshot") == 0) return Algo::OneShot;
  if (strcmp(s, "2stage") == 0 || strcmp(s, "twoshot") == 0) return Algo::TwoShot;
  if (strcmp(s, "push") == 0 || strcmp(s, "pushshot") == 0) return Algo::Push;
  fprintf(stderr, "Invalid algo '%s'. Valid values: auto, oneshot, twoshot, push\n", s);
  exit(1);
}

Algo resolve_algo(Algo requested, int ngpus, size_t bytes) {
  if (requested != Algo::Auto) return requested;
  if (ngpus == 2) return Algo::OneShot;
  if ((ngpus <= 4 && bytes < kVllmSmallThreshold) ||
      (ngpus <= 8 && bytes < kVllmLargeThreshold)) {
    return Algo::OneShot;
  }
  return Algo::TwoShot;
}

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

template <int ngpus>
void launch_reduce(Algo algo, RankData* rd, RankSignals sg, Signal* signal,
                   const half* input, half* output, int rank, int packed_size,
                   int nblocks, size_t buffer_bytes, cudaStream_t stream) {
  if (algo == Algo::TwoShot) {
    cross_device_reduce_2stage<half, ngpus>
        <<<nblocks, 512, 0, stream>>>(rd, sg, signal, output, rank, packed_size);
  } else if (algo == Algo::Push) {
    cross_device_reduce_push_safe<half, ngpus>
        <<<nblocks, 512, 0, stream>>>(rd, sg, signal, input, output, rank,
                                      packed_size, buffer_bytes);
  } else {
    cross_device_reduce_1stage<half, ngpus>
        <<<nblocks, 512, 0, stream>>>(rd, sg, signal, output, rank, packed_size);
  }
}

struct GPUState {
  half* input;
  half* output;
  void* push_buffer;
  Signal* signal;
  RankData* rank_data;
  cudaStream_t stream;
  cudaGraphExec_t graph_exec;
  cudaEvent_t start;
  cudaEvent_t stop;
};

struct BenchArgs {
  int rank;
  int ngpus;
  int warmup;
  int iters;
  int repeats;
  int direct_warmup;
  bool verify;
  Algo requested_algo;
  size_t bytes;
  GPUState* states;
  RankSignals sg;
  pthread_barrier_t* barrier;
  float times[4096];
};

template <int ngpus>
void* bench_thread(void* arg) {
  auto* a = reinterpret_cast<BenchArgs*>(arg);
  int rank = a->rank;
  CHECK_CUDA(cudaSetDevice(rank));
  cudaStream_t stream = a->states[rank].stream;
  Signal* signal = a->states[rank].signal;
  RankData* rd = a->states[rank].rank_data;
  half* input = a->states[rank].input;
  half* output = a->states[rank].output;
  Algo algo = resolve_algo(a->requested_algo, a->ngpus, a->bytes);

  int nelems = static_cast<int>(a->bytes / sizeof(half));
  int packed_size = static_cast<int>(a->bytes / sizeof(typename packed_t<half>::P));
  int nblocks = std::min((packed_size + 511) / 512, kMaxBlocks);
  if (nblocks < 1) nblocks = 1;

  if (a->verify) {
    int fill_blocks = std::min((nelems + 255) / 256, 1024);
    fill_half_kernel<<<fill_blocks, 256, 0, stream>>>(input, nelems, float(rank + 1));
    CHECK_CUDA(cudaMemsetAsync(output, 0, a->bytes, stream));
    CHECK_CUDA(cudaMemsetAsync(signal, 0, sizeof(Signal) + a->bytes, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    pthread_barrier_wait(a->barrier);
    if (algo == Algo::Push) {
      CHECK_CUDA(cudaMemsetAsync(a->states[rank].push_buffer, 0,
                                 2 * a->ngpus * a->bytes, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
      pthread_barrier_wait(a->barrier);
    }
    launch_reduce<ngpus>(algo, rd, a->sg, signal, input, output, rank,
                         packed_size, nblocks, a->bytes, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    pthread_barrier_wait(a->barrier);

    half host[16];
    size_t check_elems = std::min<size_t>(16, nelems);
    CHECK_CUDA(cudaMemcpy(host, output, check_elems * sizeof(half),
                          cudaMemcpyDeviceToHost));
    float expected = float(a->ngpus * (a->ngpus + 1) / 2);
    for (size_t i = 0; i < check_elems; i++) {
      float got = __half2float(host[i]);
      if (got != expected) {
        fprintf(stderr,
                "Verify failed: rank=%d elem=%zu got=%g expected=%g algo=%s "
                "bytes=%zu\n",
                rank, i, got, expected, algo_name(algo), a->bytes);
        exit(2);
      }
    }
    pthread_barrier_wait(a->barrier);
  }

  if (a->verify) {
    int fill_blocks = std::min((nelems + 255) / 256, 1024);
    fill_half_kernel<<<fill_blocks, 256, 0, stream>>>(input, nelems,
                                                       float(rank + 1));
  } else {
    CHECK_CUDA(cudaMemset(input, 1, a->bytes));
  }
  CHECK_CUDA(cudaMemset(output, 0, a->bytes));
  CHECK_CUDA(cudaMemset(signal, 0, sizeof(Signal) + a->bytes));
  if (algo == Algo::Push) {
    CHECK_CUDA(cudaMemset(a->states[rank].push_buffer, 0,
                          2 * a->ngpus * a->bytes));
  }
  pthread_barrier_wait(a->barrier);

  for (int w = 0; w < a->direct_warmup; w++) {
    launch_reduce<ngpus>(algo, rd, a->sg, signal, input, output, rank,
                         packed_size, nblocks, a->bytes, stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  pthread_barrier_wait(a->barrier);

  cudaGraph_t graph;
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int rep = 0; rep < a->repeats; rep++) {
    launch_reduce<ngpus>(algo, rd, a->sg, signal, input, output, rank,
                         packed_size, nblocks, a->bytes, stream);
  }
  CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
  CHECK_CUDA(cudaGraphInstantiate(&a->states[rank].graph_exec, graph, nullptr,
                                  nullptr, 0));
  CHECK_CUDA(cudaGraphDestroy(graph));

  for (int w = 0; w < a->warmup; w++) {
    pthread_barrier_wait(a->barrier);
    CHECK_CUDA(cudaGraphLaunch(a->states[rank].graph_exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  pthread_barrier_wait(a->barrier);

  for (int it = 0; it < a->iters; it++) {
    pthread_barrier_wait(a->barrier);
    CHECK_CUDA(cudaEventRecord(a->states[rank].start, stream));
    CHECK_CUDA(cudaGraphLaunch(a->states[rank].graph_exec, stream));
    CHECK_CUDA(cudaEventRecord(a->states[rank].stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, a->states[rank].start,
                                    a->states[rank].stop));
    a->times[it] = ms * 1000.0f / a->repeats;
  }

  if (a->verify) {
    pthread_barrier_wait(a->barrier);
    half host[16];
    size_t check_elems = std::min<size_t>(16, nelems);
    CHECK_CUDA(cudaMemcpy(host, output, check_elems * sizeof(half),
                          cudaMemcpyDeviceToHost));
    float expected = float(a->ngpus * (a->ngpus + 1) / 2);
    for (size_t i = 0; i < check_elems; i++) {
      float got = __half2float(host[i]);
      if (got != expected) {
        fprintf(stderr,
                "Graph verify failed: rank=%d elem=%zu got=%g expected=%g "
                "algo=%s bytes=%zu repeats=%d\n",
                rank, i, got, expected, algo_name(algo), a->bytes,
                a->repeats);
        exit(2);
      }
    }
    pthread_barrier_wait(a->barrier);
  }
  return nullptr;
}

void bench_size(GPUState* states, int ngpus, size_t bytes, int warmup, int iters,
                int repeats, int direct_warmup, Algo requested_algo,
                bool verify, const char* label) {
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, nullptr, ngpus);

  RankSignals sg;
  for (int i = 0; i < ngpus; i++) sg.signals[i] = states[i].signal;

  BenchArgs args[8] = {};
  pthread_t threads[8];
  for (int r = 0; r < ngpus; r++) {
    args[r].rank = r;
    args[r].ngpus = ngpus;
    args[r].warmup = warmup;
    args[r].iters = iters;
    args[r].repeats = repeats;
    args[r].direct_warmup = direct_warmup;
    args[r].verify = verify;
    args[r].requested_algo = requested_algo;
    args[r].bytes = bytes;
    args[r].states = states;
    args[r].sg = sg;
    args[r].barrier = &barrier;
    if (ngpus == 2) pthread_create(&threads[r], nullptr, bench_thread<2>, &args[r]);
    else if (ngpus == 4) pthread_create(&threads[r], nullptr, bench_thread<4>, &args[r]);
    else if (ngpus == 6) pthread_create(&threads[r], nullptr, bench_thread<6>, &args[r]);
    else if (ngpus == 8) pthread_create(&threads[r], nullptr, bench_thread<8>, &args[r]);
  }
  for (int r = 0; r < ngpus; r++) pthread_join(threads[r], nullptr);
  pthread_barrier_destroy(&barrier);

  for (int r = 0; r < ngpus; r++) {
    CHECK_CUDA(cudaSetDevice(r));
    if (states[r].graph_exec) {
      CHECK_CUDA(cudaGraphExecDestroy(states[r].graph_exec));
      states[r].graph_exec = nullptr;
    }
  }

  float max_times[4096];
  for (int it = 0; it < iters; it++) {
    float mx = 0;
    for (int r = 0; r < ngpus; r++) mx = std::max(mx, args[r].times[it]);
    max_times[it] = mx;
  }
  std::sort(max_times, max_times + iters);
  float median = max_times[iters / 2];
  float p99 = max_times[std::min(iters - 1, static_cast<int>(iters * 0.99f))];
  float mean = 0;
  for (int i = 0; i < iters; i++) mean += max_times[i];
  mean /= iters;

  const char* unit = "B";
  float display_size = bytes;
  if (bytes >= 1024 * 1024) {
    display_size = bytes / (1024.0f * 1024.0f);
    unit = "MB";
  } else if (bytes >= 1024) {
    display_size = bytes / 1024.0f;
    unit = "KB";
  }
  Algo resolved = resolve_algo(requested_algo, ngpus, bytes);
  const char* out_label = label ? label : "-";
  if (verify) printf("%16s %8.0f %-2s %12s\n", out_label, display_size, unit, "verify OK");
  printf("%16s %8.0f %-2s %12s %12.1f %12.1f %12.1f\n", out_label,
         display_size, unit, algo_name(resolved), median, mean, p99);
}

void setup_gpus(GPUState* states, int ngpus, size_t max_bytes,
                bool need_push) {
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int j = 0; j < ngpus; j++) {
      if (i == j) continue;
      int can_access = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, i, j));
      if (can_access) {
        cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
          fprintf(stderr, "Peer access %d -> %d failed: %s\n", i, j,
                  cudaGetErrorString(e));
          exit(1);
        }
        cudaGetLastError();
      }
    }
  }

  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&states[i].input, max_bytes));
    CHECK_CUDA(cudaMalloc(&states[i].output, max_bytes));
    if (need_push) {
      CHECK_CUDA(cudaMalloc(&states[i].push_buffer, 2 * ngpus * max_bytes));
      CHECK_CUDA(cudaMemset(states[i].push_buffer, 0, 2 * ngpus * max_bytes));
    }
    CHECK_CUDA(cudaMalloc(&states[i].signal, sizeof(Signal) + max_bytes));
    CHECK_CUDA(cudaMalloc(&states[i].rank_data, sizeof(RankData)));
    CHECK_CUDA(cudaMemset(states[i].signal, 0, sizeof(Signal) + max_bytes));
    CHECK_CUDA(cudaStreamCreate(&states[i].stream));
    CHECK_CUDA(cudaEventCreate(&states[i].start));
    CHECK_CUDA(cudaEventCreate(&states[i].stop));
  }

  for (int i = 0; i < ngpus; i++) {
    RankData rd = {};
    for (int j = 0; j < ngpus; j++) {
      rd.ptrs[j] = states[j].input;
      rd.push_buffers[j] = states[j].push_buffer;
    }
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpy(states[i].rank_data, &rd, sizeof(RankData),
                          cudaMemcpyHostToDevice));
  }
}

void cleanup_gpus(GPUState* states, int ngpus) {
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    if (states[i].graph_exec) CHECK_CUDA(cudaGraphExecDestroy(states[i].graph_exec));
    CHECK_CUDA(cudaFree(states[i].input));
    CHECK_CUDA(cudaFree(states[i].output));
    if (states[i].push_buffer) CHECK_CUDA(cudaFree(states[i].push_buffer));
    CHECK_CUDA(cudaFree(states[i].signal));
    CHECK_CUDA(cudaFree(states[i].rank_data));
    CHECK_CUDA(cudaStreamDestroy(states[i].stream));
    CHECK_CUDA(cudaEventDestroy(states[i].start));
    CHECK_CUDA(cudaEventDestroy(states[i].stop));
  }
}

int main(int argc, char** argv) {
  if (argc > 1 &&
      (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 ||
       strcmp(argv[1], "help") == 0)) {
    print_usage(argv[0]);
    return 0;
  }

  int ngpus = argc > 1 ? atoi(argv[1]) : 2;
  int warmup = argc > 2 ? atoi(argv[2]) : 20;
  int iters = argc > 3 ? atoi(argv[3]) : 50;
  size_t single_size = argc > 4 ? strtoull(argv[4], nullptr, 10) : 0;
  int repeats = argc > 5 ? atoi(argv[5]) : 20;
  int direct_warmup = argc > 6 ? atoi(argv[6]) : 0;
  Algo requested_algo = argc > 7 ? parse_algo(argv[7]) : Algo::Auto;
  bool verify = argc > 8 ? atoi(argv[8]) != 0 : true;
  const char* label = parse_label_arg(argc, argv, 9);
  if (label && label[0] == '\0') label = nullptr;

  if (ngpus != 2 && ngpus != 4 && ngpus != 6 && ngpus != 8) {
    fprintf(stderr, "Only 2, 4, 6, 8 GPUs supported\n");
    return 1;
  }
  if (iters <= 0 || iters > 4096) {
    fprintf(stderr, "iters must be in [1, 4096]\n");
    return 1;
  }
  if (repeats <= 0) {
    fprintf(stderr, "repeats must be positive\n");
    return 1;
  }

  size_t default_sizes[] = {1024, 4096, 6144, 8192, 16384, 32768, 65536,
                            131072, 262144, 524288, 1048576};
  size_t single_sizes[] = {single_size};
  size_t* sizes = single_size ? single_sizes : default_sizes;
  int nsizes = single_size ? 1 : int(sizeof(default_sizes) / sizeof(default_sizes[0]));
  for (int s = 0; s < nsizes; s++) {
    if (sizes[s] == 0 || sizes[s] % 16 != 0) {
      fprintf(stderr, "size_bytes must be a positive multiple of 16, got %zu\n",
              sizes[s]);
      return 1;
    }
    if (requested_algo == Algo::Push && sizes[s] > kPushMaxBytes) {
      fprintf(stderr, "push supports up to %zu bytes in this benchmark, got %zu\n",
              kPushMaxBytes, sizes[s]);
      return 1;
    }
  }

  GPUState states[8] = {};
  setup_gpus(states, ngpus, sizes[nsizes - 1], requested_algo == Algo::Push);

  printf("vLLM Custom All-Reduce Standalone [out-of-place + CUDA Graph]\n");
  printf("  GPUs: %d, Requested Algo: %s, Direct warmup: %d, Graph warmup: %d, "
         "Iters: %d, Repeats/graph: %d, Verify: %s, Label: %s\n\n",
         ngpus, algo_name(requested_algo), direct_warmup, warmup, iters,
         repeats, verify ? "on" : "off", label ? label : "-");
  printf("%16s %8s    %12s %12s %12s %12s\n", "Label", "Size", "Algo",
         "Median (us)", "Mean (us)", "P99 (us)");
  printf("------------------------------------------------------------------------------------\n");

  for (int s = 0; s < nsizes; s++) {
    bench_size(states, ngpus, sizes[s], warmup, iters, repeats, direct_warmup,
               requested_algo, verify, label);
  }

  cleanup_gpus(states, ngpus);
  return 0;
}
