/*
 * Push-style PCIe oneshot all-reduce breakdown.
 *
 * This is an instrumented variant of the extracted push kernel in
 * bench_oneshot.cu.  It uses clock64() inside the kernel to attribute the
 * hot path into:
 *
 *   load_clear, store_local, store_remote, poll, reduce, write_result, clear
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_120 bench_push_breakdown.cu -o bench_push_breakdown -lpthread
 *
 * Run:
 *   ./bench_push_breakdown [ngpus] [size_bytes] [warmup] [iters]
 */

#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
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

constexpr int kMaxBlocks = 36;
constexpr int kMaxIters = 2048;
constexpr int kPhaseCount = 9;

struct Signal {
  alignas(128) uint32_t self_counter[kMaxBlocks][8];
};

struct __align__(16) RankData {
  void* __restrict__ push_buffers[8];
};

struct __align__(16) HalfPack {
  half data[8];
};

struct PushTiming {
  unsigned long long total;
  unsigned long long load_clear;
  unsigned long long store_local;
  unsigned long long store_remote;
  unsigned long long poll;
  unsigned long long reduce;
  unsigned long long write_result;
  unsigned long long clear;
  unsigned long long sync_epoch;
  unsigned long long poll_iters;
  unsigned int active_threads;
};

#define DINLINE __device__ __forceinline__

DINLINE bool is_pos_zero(half v) {
  return *reinterpret_cast<uint16_t*>(&v) == 0x0000u;
}

DINLINE void clear_pos_zero(half& v) {
  uint16_t* bits = reinterpret_cast<uint16_t*>(&v);
  if (*bits == 0x0000u) *bits = 0x8000u;
}

DINLINE void clear_pos_zero_pack(HalfPack& v) {
  for (int i = 0; i < 8; i++) clear_pos_zero(v.data[i]);
}

DINLINE bool has_pos_zero_pack(const HalfPack& v) {
  bool found = false;
  for (int i = 0; i < 8; i++) found |= is_pos_zero(v.data[i]);
  return found;
}

DINLINE HalfPack make_pos_zero_pack() {
  HalfPack v;
  for (int i = 0; i < 8; i++) v.data[i] = half{};
  return v;
}

DINLINE void ld_global_volatile_16B(HalfPack& x, const HalfPack* addr) {
  uint4 val;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<HalfPack*>(&val);
}

DINLINE void st_global_volatile_16B(const HalfPack& x, HalfPack* addr) {
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::
                   "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

template <int ngpus>
DINLINE HalfPack reduce_storage(HalfPack (&storage)[ngpus]) {
  HalfPack out;
  for (int lane = 0; lane < 8; lane++) {
    float sum = __half2float(storage[0].data[lane]);
    for (int r = 1; r < ngpus; r++) sum += __half2float(storage[r].data[lane]);
    out.data[lane] = __float2half(sum);
  }
  return out;
}

template <int ngpus>
__global__ void __launch_bounds__(512, 1) push_breakdown_kernel(
    RankData* _dp, Signal* self_sg, half* __restrict__ data, int rank,
    int packed_size, size_t buffer_bytes, PushTiming* timing) {
  auto dp = *_dp;
  uint32_t epoch = self_sg->self_counter[blockIdx.x][0] & 1u;
  size_t epoch_offset = epoch * ngpus * buffer_bytes;

  HalfPack* push_bufs[ngpus];
  HalfPack* poll_bufs[ngpus];
  for (int i = 0; i < ngpus; i++) {
    push_bufs[i] = reinterpret_cast<HalfPack*>(
        reinterpret_cast<char*>(dp.push_buffers[i]) + epoch_offset + rank * buffer_bytes);
    poll_bufs[i] = reinterpret_cast<HalfPack*>(
        reinterpret_cast<char*>(dp.push_buffers[rank]) + epoch_offset + i * buffer_bytes);
  }

  unsigned long long t_total0 = clock64();
  unsigned long long t_load_clear = 0;
  unsigned long long t_store_local = 0;
  unsigned long long t_store_remote = 0;
  unsigned long long t_poll = 0;
  unsigned long long t_reduce = 0;
  unsigned long long t_write_result = 0;
  unsigned long long t_clear = 0;
  unsigned long long poll_iters = 0;
  unsigned long long active = 0;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packed_size;
       idx += gridDim.x * blockDim.x) {
    active = 1;
    unsigned long long t0 = clock64();
    HalfPack vec = reinterpret_cast<const HalfPack*>(data)[idx];
    clear_pos_zero_pack(vec);
    t_load_clear += clock64() - t0;

    for (int i = 0; i < ngpus; i++) {
      t0 = clock64();
      st_global_volatile_16B(vec, push_bufs[i] + idx);
      unsigned long long dt = clock64() - t0;
      if (i == rank) t_store_local += dt;
      else t_store_remote += dt;
    }
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packed_size;
       idx += gridDim.x * blockDim.x) {
    HalfPack storage[ngpus];
    unsigned long long t0 = clock64();
    while (true) {
      bool has_pos_zero = false;
      poll_iters++;
      for (int i = 0; i < ngpus; i++) {
        ld_global_volatile_16B(storage[i], poll_bufs[i] + idx);
        has_pos_zero |= has_pos_zero_pack(storage[i]);
      }
      if (!has_pos_zero) break;
    }
    t_poll += clock64() - t0;

    t0 = clock64();
    HalfPack reduced = reduce_storage<ngpus>(storage);
    t_reduce += clock64() - t0;

    t0 = clock64();
    reinterpret_cast<HalfPack*>(data)[idx] = reduced;
    t_write_result += clock64() - t0;

    HalfPack zeros = make_pos_zero_pack();
    t0 = clock64();
    for (int i = 0; i < ngpus; i++) st_global_volatile_16B(zeros, poll_bufs[i] + idx);
    t_clear += clock64() - t0;
  }

  unsigned long long t_sync0 = clock64();
  __syncthreads();
  if (threadIdx.x == 0) self_sg->self_counter[blockIdx.x][0] = (epoch + 1u) & 1u;
  unsigned long long t_end = clock64();

  __shared__ unsigned long long sh_total[512];
  __shared__ unsigned long long sh_load_clear[512];
  __shared__ unsigned long long sh_store_local[512];
  __shared__ unsigned long long sh_store_remote[512];
  __shared__ unsigned long long sh_poll[512];
  __shared__ unsigned long long sh_reduce[512];
  __shared__ unsigned long long sh_write_result[512];
  __shared__ unsigned long long sh_clear[512];
  __shared__ unsigned long long sh_sync_epoch[512];
  __shared__ unsigned long long sh_poll_iters[512];
  __shared__ unsigned long long sh_active[512];

  int tid = threadIdx.x;
  sh_total[tid] = t_end - t_total0;
  sh_load_clear[tid] = t_load_clear;
  sh_store_local[tid] = t_store_local;
  sh_store_remote[tid] = t_store_remote;
  sh_poll[tid] = t_poll;
  sh_reduce[tid] = t_reduce;
  sh_write_result[tid] = t_write_result;
  sh_clear[tid] = t_clear;
  sh_sync_epoch[tid] = t_end - t_sync0;
  sh_poll_iters[tid] = poll_iters;
  sh_active[tid] = active;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sh_total[tid] = max(sh_total[tid], sh_total[tid + offset]);
      sh_load_clear[tid] = max(sh_load_clear[tid], sh_load_clear[tid + offset]);
      sh_store_local[tid] = max(sh_store_local[tid], sh_store_local[tid + offset]);
      sh_store_remote[tid] = max(sh_store_remote[tid], sh_store_remote[tid + offset]);
      sh_poll[tid] = max(sh_poll[tid], sh_poll[tid + offset]);
      sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + offset]);
      sh_write_result[tid] = max(sh_write_result[tid], sh_write_result[tid + offset]);
      sh_clear[tid] = max(sh_clear[tid], sh_clear[tid + offset]);
      sh_sync_epoch[tid] = max(sh_sync_epoch[tid], sh_sync_epoch[tid + offset]);
      sh_poll_iters[tid] += sh_poll_iters[tid + offset];
      sh_active[tid] += sh_active[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    PushTiming* out = &timing[blockIdx.x];
    out->total = sh_total[0];
    out->load_clear = sh_load_clear[0];
    out->store_local = sh_store_local[0];
    out->store_remote = sh_store_remote[0];
    out->poll = sh_poll[0];
    out->reduce = sh_reduce[0];
    out->write_result = sh_write_result[0];
    out->clear = sh_clear[0];
    out->sync_epoch = sh_sync_epoch[0];
    out->poll_iters = sh_poll_iters[0];
    out->active_threads = static_cast<unsigned int>(sh_active[0]);
  }
}

__global__ void fill_half_kernel(half* data, int nelems, float value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  half v = __float2half(value);
  for (int i = tid; i < nelems; i += stride) data[i] = v;
}

struct GPUState {
  half* data;
  void* push_buffer;
  Signal* signal;
  RankData* rank_data;
  PushTiming* timing;
  cudaStream_t stream;
  cudaGraphExec_t graph_exec;
  cudaEvent_t start;
  cudaEvent_t stop;
  float event_us;
  PushTiming host_timing[kMaxBlocks];
};

struct BenchArgs {
  int rank;
  int ngpus;
  int warmup;
  int iters;
  size_t bytes;
  GPUState* states;
  pthread_barrier_t* barrier;
  unsigned long long samples[kPhaseCount][kMaxIters];
  double poll_iter_samples[kMaxIters];
  float event_samples[kMaxIters];
};

template <int ngpus>
void launch_kernel(GPUState& s, int rank, int packed_size, size_t bytes) {
  int nblocks = std::min((packed_size + 511) / 512, kMaxBlocks);
  if (nblocks < 1) nblocks = 1;
  push_breakdown_kernel<ngpus><<<nblocks, 512, 0, s.stream>>>(
      s.rank_data, s.signal, s.data, rank, packed_size, bytes, s.timing);
}

static unsigned long long max_phase(const PushTiming* timing, int nblocks,
                                    unsigned long long PushTiming::*field) {
  unsigned long long v = 0;
  for (int b = 0; b < nblocks; b++) v = std::max(v, timing[b].*field);
  return v;
}

static unsigned long long sum_phase(const PushTiming* timing, int nblocks,
                                    unsigned long long PushTiming::*field) {
  unsigned long long v = 0;
  for (int b = 0; b < nblocks; b++) v += timing[b].*field;
  return v;
}

template <int ngpus>
void* bench_thread(void* arg) {
  auto* a = reinterpret_cast<BenchArgs*>(arg);
  int rank = a->rank;
  CHECK_CUDA(cudaSetDevice(rank));
  GPUState& s = a->states[rank];

  int nelems = a->bytes / sizeof(half);
  int packed_size = a->bytes / sizeof(HalfPack);
  int fill_blocks = std::min((nelems + 255) / 256, 1024);

  fill_half_kernel<<<fill_blocks, 256, 0, s.stream>>>(s.data, nelems, float(rank + 1));
  CHECK_CUDA(cudaMemsetAsync(s.push_buffer, 0, 2 * a->ngpus * a->bytes, s.stream));
  CHECK_CUDA(cudaMemsetAsync(s.signal, 0, sizeof(Signal), s.stream));
  CHECK_CUDA(cudaStreamSynchronize(s.stream));
  pthread_barrier_wait(a->barrier);

  cudaGraph_t graph;
  CHECK_CUDA(cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeThreadLocal));
  CHECK_CUDA(cudaMemsetAsync(s.timing, 0, kMaxBlocks * sizeof(PushTiming), s.stream));
  launch_kernel<ngpus>(s, rank, packed_size, a->bytes);
  CHECK_CUDA(cudaStreamEndCapture(s.stream, &graph));
  CHECK_CUDA(cudaGraphInstantiate(&s.graph_exec, graph, nullptr, nullptr, 0));
  CHECK_CUDA(cudaGraphDestroy(graph));

  for (int w = 0; w < a->warmup; w++) {
    CHECK_CUDA(cudaGraphLaunch(s.graph_exec, s.stream));
    CHECK_CUDA(cudaStreamSynchronize(s.stream));
    pthread_barrier_wait(a->barrier);
  }

  for (int it = 0; it < a->iters; it++) {
    pthread_barrier_wait(a->barrier);
    CHECK_CUDA(cudaEventRecord(s.start, s.stream));
    CHECK_CUDA(cudaGraphLaunch(s.graph_exec, s.stream));
    CHECK_CUDA(cudaEventRecord(s.stop, s.stream));
    CHECK_CUDA(cudaStreamSynchronize(s.stream));
    CHECK_CUDA(cudaEventElapsedTime(&s.event_us, s.start, s.stop));
    a->event_samples[it] = s.event_us * 1000.0f;
    CHECK_CUDA(cudaMemcpy(s.host_timing, s.timing, kMaxBlocks * sizeof(PushTiming),
                          cudaMemcpyDeviceToHost));

    unsigned long long PushTiming::*fields[] = {
        &PushTiming::total, &PushTiming::load_clear, &PushTiming::store_local,
        &PushTiming::store_remote, &PushTiming::poll, &PushTiming::reduce,
        &PushTiming::write_result, &PushTiming::clear, &PushTiming::sync_epoch};
    for (int p = 0; p < kPhaseCount; p++) {
      a->samples[p][it] = max_phase(s.host_timing, kMaxBlocks, fields[p]);
    }
    unsigned long long poll_iters = sum_phase(s.host_timing, kMaxBlocks, &PushTiming::poll_iters);
    unsigned int active_threads = 0;
    for (int b = 0; b < kMaxBlocks; b++) active_threads += s.host_timing[b].active_threads;
    a->poll_iter_samples[it] = active_threads ? double(poll_iters) / active_threads : 0.0;
  }
  pthread_barrier_wait(a->barrier);
  return nullptr;
}

int main(int argc, char** argv) {
  int ngpus = argc > 1 ? atoi(argv[1]) : 2;
  size_t bytes = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1024;
  int warmup = argc > 3 ? atoi(argv[3]) : 20;
  int iters = argc > 4 ? atoi(argv[4]) : 50;

  if (ngpus != 2 && ngpus != 4) {
    fprintf(stderr, "Only 2 or 4 GPUs supported\n");
    return 1;
  }
  if (bytes == 0 || bytes % sizeof(HalfPack) != 0) {
    fprintf(stderr, "size_bytes must be a positive multiple of 16\n");
    return 1;
  }
  if (iters <= 0 || iters > kMaxIters) {
    fprintf(stderr, "iters must be in [1, %d]\n", kMaxIters);
    return 1;
  }

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  double cycles_per_us = prop.clockRate / 1000.0;

  GPUState states[8] = {};
  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int j = 0; j < ngpus; j++) {
      if (i == j) continue;
      int can_access = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, i, j));
      if (can_access) {
        cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
          fprintf(stderr, "Peer access %d -> %d failed: %s\n", i, j, cudaGetErrorString(e));
          return 1;
        }
        cudaGetLastError();
      }
    }
  }

  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&states[i].data, bytes));
    CHECK_CUDA(cudaMalloc(&states[i].push_buffer, 2 * ngpus * bytes));
    CHECK_CUDA(cudaMalloc(&states[i].signal, sizeof(Signal)));
    CHECK_CUDA(cudaMalloc(&states[i].rank_data, sizeof(RankData)));
    CHECK_CUDA(cudaMalloc(&states[i].timing, kMaxBlocks * sizeof(PushTiming)));
    CHECK_CUDA(cudaStreamCreate(&states[i].stream));
    CHECK_CUDA(cudaEventCreate(&states[i].start));
    CHECK_CUDA(cudaEventCreate(&states[i].stop));
  }

  for (int i = 0; i < ngpus; i++) {
    RankData rd = {};
    for (int j = 0; j < ngpus; j++) rd.push_buffers[j] = states[j].push_buffer;
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpy(states[i].rank_data, &rd, sizeof(RankData), cudaMemcpyHostToDevice));
  }

  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, nullptr, ngpus);
  BenchArgs args[8];
  pthread_t threads[8];
  for (int r = 0; r < ngpus; r++) {
    args[r] = {};
    args[r].rank = r;
    args[r].ngpus = ngpus;
    args[r].warmup = warmup;
    args[r].iters = iters;
    args[r].bytes = bytes;
    args[r].states = states;
    args[r].barrier = &barrier;
    if (ngpus == 2) pthread_create(&threads[r], nullptr, bench_thread<2>, &args[r]);
    else pthread_create(&threads[r], nullptr, bench_thread<4>, &args[r]);
  }
  for (int r = 0; r < ngpus; r++) pthread_join(threads[r], nullptr);
  pthread_barrier_destroy(&barrier);

  int packed_size = bytes / sizeof(HalfPack);
  int nblocks = std::min((packed_size + 511) / 512, kMaxBlocks);
  if (nblocks < 1) nblocks = 1;

  printf("Push Breakdown\n");
  printf("  GPUs: %d, Bytes: %zu, Blocks: %d, Warmup: %d, Iters: %d, cycles/us: %.1f\n\n",
         ngpus, bytes, nblocks, warmup, iters, cycles_per_us);

  float event_samples[kMaxIters];
  for (int it = 0; it < iters; it++) {
    float max_event = 0;
    for (int r = 0; r < ngpus; r++) max_event = std::max(max_event, args[r].event_samples[it]);
    event_samples[it] = max_event;
  }
  std::sort(event_samples, event_samples + iters);
  printf("Graph event total median, max rank: %.3f us\n\n", event_samples[iters / 2]);

  printf("%-16s %12s %12s %12s\n", "Phase", "cycles", "us", "rank");
  printf("--------------------------------------------------------\n");
  const char* names[] = {
      "total", "load_clear", "store_local", "store_remote", "poll",
      "reduce", "write_result", "clear", "sync_epoch"};
  for (int p = 0; p < 9; p++) {
    unsigned long long phase_samples[kMaxIters];
    for (int it = 0; it < iters; it++) {
      unsigned long long best = 0;
      for (int r = 0; r < ngpus; r++) {
        unsigned long long v = args[r].samples[p][it];
        if (v > best) best = v;
      }
      phase_samples[it] = best;
    }
    std::sort(phase_samples, phase_samples + iters);
    unsigned long long median = phase_samples[iters / 2];
    int median_rank = 0;
    for (int it = 0; it < iters; it++) {
      unsigned long long best = 0;
      for (int r = 0; r < ngpus; r++) best = std::max(best, args[r].samples[p][it]);
      if (best == median) {
        for (int r = 0; r < ngpus; r++) {
          if (args[r].samples[p][it] == median) {
            median_rank = r;
            break;
          }
        }
        break;
      }
    }
    printf("%-16s %12llu %12.3f %12d\n", names[p], median,
           median / cycles_per_us, median_rank);
  }

  double poll_iter_samples[kMaxIters];
  for (int it = 0; it < iters; it++) {
    double best = 0.0;
    for (int r = 0; r < ngpus; r++) best = std::max(best, args[r].poll_iter_samples[it]);
    poll_iter_samples[it] = best;
  }
  std::sort(poll_iter_samples, poll_iter_samples + iters);
  printf("\nMedian poll iterations per active thread: %.2f\n", poll_iter_samples[iters / 2]);

  for (int i = 0; i < ngpus; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(states[i].data));
    CHECK_CUDA(cudaFree(states[i].push_buffer));
    CHECK_CUDA(cudaFree(states[i].signal));
    CHECK_CUDA(cudaFree(states[i].rank_data));
    CHECK_CUDA(cudaFree(states[i].timing));
    CHECK_CUDA(cudaGraphExecDestroy(states[i].graph_exec));
    CHECK_CUDA(cudaStreamDestroy(states[i].stream));
    CHECK_CUDA(cudaEventDestroy(states[i].start));
    CHECK_CUDA(cudaEventDestroy(states[i].stop));
  }
  return 0;
}
