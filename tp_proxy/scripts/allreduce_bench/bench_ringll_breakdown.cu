/*
 * Ring LL AllReduce breakdown: extract NCCL's Ring LL logic with clock64() timing.
 * Measures each phase of the Ring LL protocol independently.
 *
 * Build: nvcc -O3 -std=c++17 -arch=sm_120 bench_ringll_breakdown.cu -o bench_ringll_breakdown -lpthread
 * Run:   ./bench_ringll_breakdown [ngpus] [size_bytes] [iters]
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// ============================================================
// NCCL LL data structures (from prims_ll.h)
// ============================================================
union ncclLLFifoLine {
  struct { uint32_t data1; uint32_t flag1; uint32_t data2; uint32_t flag2; };
  uint4 i4;
};

#define NCCL_LL_FLAG(step) ((uint32_t)(step) + 1)

// Timing results (per-thread-block, written from GPU)
struct TimingResult {
  uint64_t total;           // total kernel time
  uint64_t wait_send;       // time in waitSend (credit check)
  uint64_t read_ll_spin;    // time spinning in readLL (flag check)
  uint64_t store_ll;        // time in storeLL (write data+flag)
  uint64_t barrier;         // time in __syncthreads barriers
  uint64_t data_load;       // time loading local data
  uint64_t reduce;          // time in reduce operation
  int n_read_spins;         // number of readLL spin iterations
};

// ============================================================
// Ring LL simulation kernel (TP=2, single channel)
// Mimics NCCL's reduce-scatter + all-gather with LL protocol
// ============================================================

// readLL: volatile 16B load, spin on flag (from NCCL prims_ll.h:89-100)
__device__ uint64_t readLL_timed(volatile ncclLLFifoLine* src, uint32_t flag,
                                  uint64_t* spin_time, int* spin_count) {
  uint32_t data1, flag1, data2, flag2;
  uint64_t t0 = clock64();
  int spins = 0;
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                 : "l"(&src->i4) : "memory");
    spins++;
  } while ((flag1 != flag) || (flag2 != flag));
  *spin_time += clock64() - t0;
  *spin_count += spins;
  return data1 + (((uint64_t)data2) << 32);
}

// storeLL: volatile 16B store (from NCCL prims_ll.h:126-128)
__device__ void storeLL(volatile ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag),
                  "r"((uint32_t)(val >> 32)), "r"(flag) : "memory");
}

// Ring LL AllReduce for TP=2
// Phase 1 (reduce-scatter): GPU sends its half to peer, receives peer's half, reduces
// Phase 2 (all-gather): GPU sends reduced half to peer, receives peer's reduced half
__global__ void ring_ll_allreduce_timed(
    const uint32_t* __restrict__ input,   // local input data
    uint32_t* __restrict__ output,         // local output
    volatile ncclLLFifoLine* send_buf,     // peer's LL receive buffer (we write here)
    volatile ncclLLFifoLine* recv_buf,     // our LL receive buffer (peer writes here)
    int n_elements,                        // total elements (uint32_t)
    int rank,
    uint32_t step_base,                    // base step counter for flag
    TimingResult* timing) {

  int tid = threadIdx.x;
  int half = n_elements / 2;
  // TP=2: rank 0 sends first half, keeps second half
  //        rank 1 sends second half, keeps first half
  int send_offset = rank == 0 ? 0 : half;
  int recv_offset = rank == 0 ? half : 0;
  int keep_offset = rank == 0 ? half : 0;

  // Each thread handles a chunk of elements
  // LL packs 2 uint32_t per line (data1 + data2)
  int lines_per_half = (half + 1) / 2;

  uint64_t t_total = clock64();
  uint64_t t_read_spin = 0, t_store = 0, t_barrier = 0, t_load = 0, t_reduce = 0;
  int n_spins = 0;

  // ====== Phase 1: Reduce-Scatter ======
  // Send our half to peer's recv_buf with LL flags
  uint32_t send_flag = NCCL_LL_FLAG(step_base);
  for (int i = tid; i < lines_per_half; i += blockDim.x) {
    int idx = send_offset + i * 2;
    uint64_t t0 = clock64();
    uint32_t d1 = idx < n_elements ? input[idx] : 0;
    uint32_t d2 = (idx+1) < n_elements ? input[idx+1] : 0;
    t_load += clock64() - t0;
    uint64_t val = d1 + (((uint64_t)d2) << 32);

    t0 = clock64();
    storeLL(&send_buf[i], val, send_flag);
    t_store += clock64() - t0;
  }

  // Barrier between send and recv
  uint64_t tb = clock64();
  __syncthreads();
  t_barrier += clock64() - tb;

  // Receive peer's half from our recv_buf, reduce with our local data
  uint32_t recv_flag = NCCL_LL_FLAG(step_base);
  for (int i = tid; i < lines_per_half; i += blockDim.x) {
    int idx = keep_offset + i * 2;
    // Read peer's data (spin on flag)
    uint64_t peer_val = readLL_timed(&recv_buf[i], recv_flag, &t_read_spin, &n_spins);
    uint32_t peer_d1 = (uint32_t)peer_val;
    uint32_t peer_d2 = (uint32_t)(peer_val >> 32);

    // Reduce with local data
    uint64_t t0 = clock64();
    uint32_t local_d1 = idx < n_elements ? input[idx] : 0;
    uint32_t local_d2 = (idx+1) < n_elements ? input[idx+1] : 0;
    uint32_t r1 = local_d1 + peer_d1;  // sum reduce
    uint32_t r2 = local_d2 + peer_d2;
    t_reduce += clock64() - t0;

    // Write reduced result locally
    if (idx < n_elements) output[idx] = r1;
    if (idx+1 < n_elements) output[idx+1] = r2;
  }

  tb = clock64();
  __syncthreads();
  t_barrier += clock64() - tb;

  // ====== Phase 2: All-Gather ======
  // Send our reduced half to peer
  uint32_t send_flag2 = NCCL_LL_FLAG(step_base + 1);
  for (int i = tid; i < lines_per_half; i += blockDim.x) {
    int idx = keep_offset + i * 2;
    uint32_t d1 = idx < n_elements ? output[idx] : 0;
    uint32_t d2 = (idx+1) < n_elements ? output[idx+1] : 0;
    uint64_t val = d1 + (((uint64_t)d2) << 32);

    uint64_t t0 = clock64();
    storeLL(&send_buf[lines_per_half + i], val, send_flag2);
    t_store += clock64() - t0;
  }

  tb = clock64();
  __syncthreads();
  t_barrier += clock64() - tb;

  // Receive peer's reduced half
  uint32_t recv_flag2 = NCCL_LL_FLAG(step_base + 1);
  for (int i = tid; i < lines_per_half; i += blockDim.x) {
    int idx = send_offset + i * 2;
    uint64_t peer_val = readLL_timed(&recv_buf[lines_per_half + i], recv_flag2,
                                      &t_read_spin, &n_spins);
    uint32_t d1 = (uint32_t)peer_val;
    uint32_t d2 = (uint32_t)(peer_val >> 32);
    if (idx < n_elements) output[idx] = d1;
    if (idx+1 < n_elements) output[idx+1] = d2;
  }

  t_total = clock64() - t_total;

  // Write timing (thread 0 only)
  if (tid == 0 && timing) {
    timing->total = t_total;
    timing->wait_send = 0;  // no waitSend in this simplified version
    timing->read_ll_spin = t_read_spin;
    timing->store_ll = t_store;
    timing->barrier = t_barrier;
    timing->data_load = t_load;
    timing->reduce = t_reduce;
    timing->n_read_spins = n_spins;
  }
}

// ============================================================
// Benchmark harness
// ============================================================
struct GPUState {
  uint32_t* data;
  uint32_t* output;
  ncclLLFifoLine* ll_send_buf;  // points to PEER's recv_buf
  ncclLLFifoLine* ll_recv_buf;  // our recv buffer
  TimingResult* timing;
  cudaStream_t stream;
  cudaEvent_t start, stop;
};

struct BenchArgs {
  int rank, ngpus, iters;
  size_t bytes;
  GPUState* states;
  pthread_barrier_t* barrier;
  TimingResult host_timing;
  float event_us;
};

void* bench_thread(void* arg) {
  BenchArgs* a = (BenchArgs*)arg;
  int rank = a->rank;
  int peer = 1 - rank;  // TP=2 only
  CHECK(cudaSetDevice(rank));

  auto& s = a->states[rank];
  int n_elem = a->bytes / sizeof(uint32_t);
  int ll_buf_size = (n_elem / 2 + 1) * 2 * sizeof(ncclLLFifoLine);  // 2 phases

  CHECK(cudaMemset(s.data, rank + 1, a->bytes));
  CHECK(cudaMemset(s.ll_recv_buf, 0, ll_buf_size));
  pthread_barrier_wait(a->barrier);

  // Warmup
  for (int w = 0; w < 20; w++) {
    pthread_barrier_wait(a->barrier);
    CHECK(cudaMemset(s.ll_recv_buf, 0, ll_buf_size));
    ring_ll_allreduce_timed<<<1, 256, 0, s.stream>>>(
        s.data, s.output,
        a->states[peer].ll_recv_buf,  // send to peer's recv buf
        s.ll_recv_buf,                 // our recv buf
        n_elem, rank, w * 2, nullptr);
    CHECK(cudaStreamSynchronize(s.stream));
  }
  pthread_barrier_wait(a->barrier);

  // Benchmark
  float total_us = 0;
  TimingResult total_timing = {};
  for (int it = 0; it < a->iters; it++) {
    CHECK(cudaMemset(s.ll_recv_buf, 0, ll_buf_size));
    pthread_barrier_wait(a->barrier);

    CHECK(cudaEventRecord(s.start, s.stream));
    ring_ll_allreduce_timed<<<1, 256, 0, s.stream>>>(
        s.data, s.output,
        a->states[peer].ll_recv_buf,
        s.ll_recv_buf,
        n_elem, rank, (20 + it) * 2, s.timing);
    CHECK(cudaEventRecord(s.stop, s.stream));
    CHECK(cudaStreamSynchronize(s.stream));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, s.start, s.stop));
    total_us += ms * 1000;

    // Read timing
    TimingResult t;
    CHECK(cudaMemcpy(&t, s.timing, sizeof(TimingResult), cudaMemcpyDeviceToHost));
    total_timing.total += t.total;
    total_timing.read_ll_spin += t.read_ll_spin;
    total_timing.store_ll += t.store_ll;
    total_timing.barrier += t.barrier;
    total_timing.data_load += t.data_load;
    total_timing.reduce += t.reduce;
    total_timing.n_read_spins += t.n_read_spins;
  }

  a->event_us = total_us / a->iters;
  a->host_timing.total = total_timing.total / a->iters;
  a->host_timing.read_ll_spin = total_timing.read_ll_spin / a->iters;
  a->host_timing.store_ll = total_timing.store_ll / a->iters;
  a->host_timing.barrier = total_timing.barrier / a->iters;
  a->host_timing.data_load = total_timing.data_load / a->iters;
  a->host_timing.reduce = total_timing.reduce / a->iters;
  a->host_timing.n_read_spins = total_timing.n_read_spins / a->iters;
  return nullptr;
}

int main(int argc, char** argv) {
  int ngpus = 2;  // TP=2 only
  size_t bytes = argc > 1 ? atoi(argv[1]) : 4096;
  int iters = argc > 2 ? atoi(argv[2]) : 100;

  // Get GPU clock rate for converting cycles to us
  int clock_khz;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  float cycles_per_us = clock_khz / 1000.0f;

  GPUState states[2];
  int n_elem = bytes / sizeof(uint32_t);
  int ll_buf_size = (n_elem / 2 + 1) * 2 * sizeof(ncclLLFifoLine);

  for (int i = 0; i < 2; i++) {
    CHECK(cudaSetDevice(i));
    int c; CHECK(cudaDeviceCanAccessPeer(&c, i, 1-i)); if(c) CHECK(cudaDeviceEnablePeerAccess(1-i, 0));
    CHECK(cudaMalloc(&states[i].data, bytes));
    CHECK(cudaMalloc(&states[i].output, bytes));
    CHECK(cudaMalloc(&states[i].ll_recv_buf, ll_buf_size));
    CHECK(cudaMalloc(&states[i].timing, sizeof(TimingResult)));
    CHECK(cudaStreamCreate(&states[i].stream));
    CHECK(cudaEventCreate(&states[i].start));
    CHECK(cudaEventCreate(&states[i].stop));
  }

  printf("Ring LL AllReduce Breakdown (TP=2)\n");
  printf("  Data: %zu bytes, Iters: %d, Clock: %.0f MHz\n\n", bytes, iters, cycles_per_us);

  pthread_barrier_t bar;
  pthread_barrier_init(&bar, nullptr, 2);

  BenchArgs args[2];
  pthread_t threads[2];
  for (int r = 0; r < 2; r++) {
    args[r] = {r, 2, iters, bytes, states, &bar, {}, 0};
    pthread_create(&threads[r], nullptr, bench_thread, &args[r]);
  }
  for (int r = 0; r < 2; r++) pthread_join(threads[r], nullptr);
  pthread_barrier_destroy(&bar);

  // Report (GPU 0)
  auto& t = args[0].host_timing;
  float total_us = t.total / cycles_per_us;
  printf("cudaEvent total: %.1f us\n", args[0].event_us);
  printf("clock64() total: %.1f us (%.0f cycles)\n\n", total_us, (float)t.total);

  printf("%-25s %8s %8s\n", "Phase", "Cycles", "us");
  printf("────────────────────────────────────────────\n");
  printf("%-25s %8llu %8.1f\n", "readLL spin (flag wait)", t.read_ll_spin, t.read_ll_spin / cycles_per_us);
  printf("%-25s %8llu %8.1f\n", "storeLL (data+flag wr)", t.store_ll, t.store_ll / cycles_per_us);
  printf("%-25s %8llu %8.1f\n", "__syncthreads barrier", t.barrier, t.barrier / cycles_per_us);
  printf("%-25s %8llu %8.1f\n", "data load (local)", t.data_load, t.data_load / cycles_per_us);
  printf("%-25s %8llu %8.1f\n", "reduce (add)", t.reduce, t.reduce / cycles_per_us);
  uint64_t accounted = t.read_ll_spin + t.store_ll + t.barrier + t.data_load + t.reduce;
  uint64_t other = t.total > accounted ? t.total - accounted : 0;
  printf("%-25s %8llu %8.1f\n", "other (setup, etc)", other, other / cycles_per_us);
  printf("────────────────────────────────────────────\n");
  printf("%-25s %8llu %8.1f\n", "TOTAL", t.total, total_us);
  printf("\nAvg readLL spins/iter: %d\n", t.n_read_spins);

  for (int i = 0; i < 2; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(states[i].data)); CHECK(cudaFree(states[i].output));
    CHECK(cudaFree(states[i].ll_recv_buf)); CHECK(cudaFree(states[i].timing));
  }
  return 0;
}
