# vLLM autoregressive decode CPU overhead report

Date: 2026-05-12 UTC

## Summary

This run profiles vLLM's autoregressive generation path after model load and warmup. The key finding is that the dominant host-side cost is not tokenization or detokenization; it is the per-decode-step control path that prepares the batch, updates KV/block metadata, launches many CUDA kernels, samples the next token, updates request state, and emits outputs.

For `concurrency=1`, each output token effectively pays for one full decode step, so CUDA launch overhead is large per output token. For `concurrency=64`, the same decode step serves 64 requests, so scheduler/KV/launch overhead is amortized: measured host CUDA launch API time drops from ~11.48 ms/output-token to ~0.17 ms/output-token.

## Environment

- GPU: NVIDIA H800 PCIe, 81,559 MiB, driver 580.82.07
- Nsight Systems: 2024.6.2
- vLLM: 0.20.1
- PyTorch: 2.11.0+cu130
- Transformers: 5.8.0
- Model: `/root/autodl-tmp/qwen35_quant_experiment/models/qwen35-a3b-r3-gptq-g32-w4a16`
- Architecture: `Qwen3_5MoeForConditionalGeneration`
- Text config: 40 layers, hidden size 2048, 16 attention heads, 2 KV heads, 256 experts, top-8 experts/token
- Quantization: compressed-tensors, W4A16 style, group size 32
- vLLM flags: `enforce_eager=True`, `language_model_only=True`, `gdn_prefill_backend=triton`
- Profiling mode: `VLLM_ENABLE_V1_MULTIPROCESSING=0`, so EngineCore runs in-process and `cProfile` can attribute Python functions inside the autoregressive window.

## Method

I used two profilers against the same steady-state generation window:

1. `cProfile` around only `llm.generate(...)`, after model load and warmup. This gives Python/vLLM function call counts, self time, and cumulative time.
2. `nsys profile` with an NVTX capture range named `vllm_profile_window`, emitted only around `llm.generate(...)`. This gives CUDA API, OS runtime, and GPU kernel summaries without model-loading noise.

Important caveat: `cProfile` cumulative time includes time spent inside child calls and can include blocking native/CUDA calls. For pure Python overhead, use self time and the smaller control-plane functions. For host CUDA launch overhead, use the `nsys` CUDA API table.

## Scenarios

| Scenario | Requests | Output tokens/request | Total output tokens | cProfile window | cProfile tok/s | nsys window | nsys tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `concurrency=1` | 8 | 32 | 256 | 23.13 s | 11.07 | 25.77 s | 9.93 |
| `concurrency=64` | 64 | 32 | 2048 | 3.49 s | 586.38 | 3.18 s | 644.79 |

## Decode Call Path

One autoregressive decode iteration follows this host-side path:

1. `Scheduler.schedule()` decides which requests get tokens this step, applies token budget and max model length limits, and asks KV managers for block allocation.
2. KV cache managers calculate/allocate slots and update block tables.
3. `GPUModelRunner.execute_model()` updates persistent batch state, prepares input ids/positions/slot mapping, builds attention metadata, and launches model forward.
4. Model forward calls Qwen3.5 layers, MoE routing, Marlin quantized linear kernels, GDN/linear attention, FlashAttention, RMSNorm, and related CUDA/Triton/custom ops.
5. `GPUModelRunner.sample_tokens()` applies logits processors and samples the next token. In these runs `temperature=0`, so it uses greedy `argmax`.
6. `Scheduler.update_from_output()` appends sampled tokens, checks stop/max-token conditions, frees finished requests/blocks.
7. `OutputProcessor.process_outputs()` updates stats, detokenizes new tokens, and creates `RequestOutput`.

Relevant source entry points:

- `vllm/v1/core/sched/scheduler.py:352:schedule`
- `vllm/v1/core/sched/scheduler.py:1303:update_from_output`
- `vllm/v1/core/kv_cache_manager.py:264:allocate_slots`
- `vllm/v1/core/kv_cache_coordinator.py:80:get_num_blocks_to_allocate`
- `vllm/v1/worker/gpu_model_runner.py:1776:_prepare_inputs`
- `vllm/v1/worker/gpu_model_runner.py:2087:_build_attention_metadata`
- `vllm/v1/worker/gpu_model_runner.py:3786:execute_model`
- `vllm/v1/worker/gpu_model_runner.py:4139:sample_tokens`
- `vllm/v1/sample/sampler.py:68:forward`
- `vllm/v1/engine/detokenizer.py:95:update`
- `vllm/v1/engine/output_processor.py:572:process_outputs`

## CPU Function Attribution

Values below are cumulative `cProfile` time in the profiled generation window. Per-token numbers divide by output tokens. The rows are not additive because cumulative times overlap through call nesting.

| Function / group | What it does | c1 calls | c1 ms/output-token | c64 calls | c64 ms/output-token |
|---|---|---:|---:|---:|---:|
| `scheduler.schedule` | Select running/waiting requests; assign per-step token budget; coordinate KV allocation | 264 | 0.205 | 33 | 0.028 |
| `scheduler.update_from_output` | Append sampled tokens; stop checks; mark/free finished requests | 264 | 0.061 | 33 | 0.016 |
| KV group total | KV/block metadata checks, allocation, skipped-block cleanup | 25,802 | 0.766 | 137,347 | 0.289 |
| `kv_cache_manager.allocate_slots` | Allocate KV cache slots for newly computed/generated tokens | 256 | 0.104 | 2,048 | 0.063 |
| `kv_cache_coordinator.get_num_blocks_to_allocate` | Compute required KV blocks across cache groups | 264 | 0.049 | 2,112 | 0.030 |
| `gpu_model_runner._prepare_inputs` | Commit block table, build request indices, positions, M-RoPE positions, copy metadata | 256 | 0.966 | 32 | 0.027 |
| `gpu_model_runner._build_attention_metadata` | Build backend attention metadata and block-table views | 256 | 0.437 | 32 | 0.007 |
| block table group | Compute slot mapping and block-table metadata | 5,752 | 0.755 | 1,664 | 0.018 |
| `gpu_model_runner.sample_tokens` | Apply grammar/logits hooks and sample next tokens | 256 | 0.372 | 32 | 0.008 |
| `sampler.forward` | Convert logits to fp32, logits processors, greedy/random sampling | 256 | 0.085 | 32 | 0.001 |
| `detokenizer.update` | Incrementally decode new token ids and check stop strings | 256 | 0.023 | 2,048 | 0.004 |
| `output_processor.process_outputs` | Stats, detokenization, `RequestOutput` creation | 264 | 0.036 | 33 | 0.007 |
| `input_processor.process_inputs` | Validate and package request inputs | 8 | 0.006 | 64 | 0.005 |
| HF tokenizer `encode` | Tokenize prompts | 8 | 0.006 | 64 | 0.004 |

Interpretation:

- The request-facing Python overheads, tokenization, output processing, and detokenization are small in this setup.
- `concurrency=64` has more absolute KV/request metadata work, but much lower per-output-token overhead because each decode iteration handles a batch.
- The most important per-token amortization is in model-runner preparation and CUDA launch overhead, not in tokenization.

## CUDA Host API / nsys

`nsys` shows the host repeatedly launching CUDA kernels and managing CUDA events. This is expected in eager mode without CUDA graphs.

| Scenario | CUDA launch API calls | Launch API total | Launch API ms/output-token | CUDA memcpy total | CUDA event/stream total |
|---|---:|---:|---:|---:|---:|
| `concurrency=1` | 540,520 | 2,939.13 ms | 11.481 | 180.77 ms | 350.34 ms |
| `concurrency=64` | 69,315 | 350.33 ms | 0.171 | 21.48 ms | 38.24 ms |

The per-decode-step launch count is similar:

- c1: 540,520 launch API calls / 256 decode steps ~= 2,111 launches/step
- c64: 69,315 launch API calls / 32 decode steps ~= 2,166 launches/step

So the host launch overhead is mostly per step, not per request. Batching 64 concurrent decodes amortizes this heavily.

Top CUDA API calls:

| Scenario | Top APIs |
|---|---|
| c1 | `cudaLaunchKernel_v7000` 494,272 calls / 2,650.79 ms; `cuLaunchKernelEx` 46,248 / 288.33 ms; `cudaMemcpyAsync_v3020` 14,424 / 180.77 ms |
| c64 | `cudaLaunchKernel_v7000` 62,294 calls / 309.64 ms; `cuLaunchKernelEx` 7,021 / 40.69 ms; `cudaMemcpyAsync_v3020` 1,803 / 21.48 ms |

Top GPU kernels are model-specific and dominated by Qwen3.5 MoE/GDN/quantized execution:

- Marlin and Marlin-MoE WNA16 kernels for quantized linear/MoE work
- `fused_recurrent_gated_delta_rule_packed_decode_kernel` for GDN/linear attention decode
- FlashAttention kernels
- RMSNorm / elementwise kernels
- MoE top-k routing and expert-token alignment kernels

## OS Runtime

`nsys` OSRT summaries are thread-summed, so totals can exceed wall time. The largest entries are waits:

- `pthread_cond_timedwait`
- `epoll_wait`
- `poll`
- `sem_wait`
- `epoll_pwait`

These are mostly idle/background waits from Python/runtime/CUDA/async infrastructure. They should not be read as CPU-busy time. Actual short CPU-active OSRT calls such as mutex lock, send/recv, read/write, and ioctl are small compared with CUDA launch/event overhead.

## What the CPU is doing in autoregressive decode

For each token step, CPU-side vLLM work is:

1. Scheduling: update per-request token counts and choose which requests advance.
2. KV metadata: calculate required blocks, allocate/free slots, update block tables and slot mappings.
3. Input prep: construct CPU numpy metadata for request indices, positions, sequence lengths, and attention metadata; copy some metadata to GPU.
4. Kernel launch orchestration: call many PyTorch/custom/Triton ops, each causing CUDA kernel launches and CUDA event/stream calls.
5. Sampling: run logits processing and greedy argmax/top-k/top-p path as configured.
6. State update: move sampled token ids into request state, stop checks, free completed requests.
7. Output path: incremental detokenization and `RequestOutput` construction.

In these measurements, tokenization/detokenization are not the bottleneck. The CPU overhead that matters most is per-step orchestration and CUDA launch/event overhead, especially at low concurrency.

## Artifacts

Scripts:

- `profiling/vllm_cpu_overhead/scripts/run_vllm_offline_case.py`
- `profiling/vllm_cpu_overhead/scripts/run_vllm_server_case.py`
- `profiling/vllm_cpu_overhead/scripts/summarize_nsys_sqlite.py`
- `profiling/vllm_cpu_overhead/scripts/parse_pyspy_raw.py`

Committed profile summaries:

- `profiling/vllm_cpu_overhead/results/offline_c1_result.json`
- `profiling/vllm_cpu_overhead/results/offline_c64_result.json`
- `profiling/vllm_cpu_overhead/results/nsys_offline_c1_result.json`
- `profiling/vllm_cpu_overhead/results/nsys_offline_c64_result.json`
- `profiling/vllm_cpu_overhead/results/nsys_offline_c1_summary.json`
- `profiling/vllm_cpu_overhead/results/nsys_offline_c64_summary.json`
- `profiling/vllm_cpu_overhead/results/focused_profile_summary.json`
- `profiling/vllm_cpu_overhead/results/focused_nsys_summary.json`

Raw local captures were generated as `.prof`, `.nsys-rep`, `.sqlite`, and stdout log files. They are intentionally not committed because the two main Nsight SQLite exports alone are hundreds of MiB; regenerate them with the commands below when needed.

## Reproduction Commands

`cProfile`, concurrency 1:

```bash
source /root/autodl-tmp/qwen35_quant_experiment/scripts/env.sh
REPO=/root/autodl-tmp/Inference-engine
PROFILE_ROOT=$REPO/profiling/vllm_cpu_overhead
mkdir -p $PROFILE_ROOT/raw
/root/autodl-tmp/qwen35-quant-venv/bin/python \
  $PROFILE_ROOT/scripts/run_vllm_offline_case.py \
  --model /root/autodl-tmp/qwen35_quant_experiment/models/qwen35-a3b-r3-gptq-g32-w4a16 \
  --concurrency 1 --total-requests 8 --max-tokens 32 \
  --max-model-len 512 --max-num-batched-tokens 512 --warmup-requests 1 \
  --result-json $PROFILE_ROOT/results/offline_c1_result.json \
  --cprofile-out $PROFILE_ROOT/raw/offline_c1.prof
```

`cProfile`, concurrency 64:

```bash
source /root/autodl-tmp/qwen35_quant_experiment/scripts/env.sh
REPO=/root/autodl-tmp/Inference-engine
PROFILE_ROOT=$REPO/profiling/vllm_cpu_overhead
mkdir -p $PROFILE_ROOT/raw
/root/autodl-tmp/qwen35-quant-venv/bin/python \
  $PROFILE_ROOT/scripts/run_vllm_offline_case.py \
  --model /root/autodl-tmp/qwen35_quant_experiment/models/qwen35-a3b-r3-gptq-g32-w4a16 \
  --concurrency 64 --total-requests 64 --max-tokens 32 \
  --max-model-len 512 --max-num-batched-tokens 4096 --warmup-requests 4 \
  --result-json $PROFILE_ROOT/results/offline_c64_result.json \
  --cprofile-out $PROFILE_ROOT/raw/offline_c64.prof
```

`nsys`, concurrency 64 example:

```bash
source /root/autodl-tmp/qwen35_quant_experiment/scripts/env.sh
REPO=/root/autodl-tmp/Inference-engine
PROFILE_ROOT=$REPO/profiling/vllm_cpu_overhead
mkdir -p $PROFILE_ROOT/raw
nsys profile \
  -o $PROFILE_ROOT/raw/nsys_offline_c64 \
  --force-overwrite=true \
  --capture-range=nvtx \
  --nvtx-capture=vllm_profile_window \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt,python-gil \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --python-sampling=true \
  --python-sampling-frequency=500 \
  --stats=false \
  /root/autodl-tmp/qwen35-quant-venv/bin/python \
    $PROFILE_ROOT/scripts/run_vllm_offline_case.py \
    --model /root/autodl-tmp/qwen35_quant_experiment/models/qwen35-a3b-r3-gptq-g32-w4a16 \
    --concurrency 64 --total-requests 64 --max-tokens 32 \
    --max-model-len 512 --max-num-batched-tokens 4096 --warmup-requests 4 \
    --result-json $PROFILE_ROOT/results/nsys_offline_c64_result.json \
    --emit-nvtx
```

## Limitations

- This report profiles the vLLM engine autoregressive path, not full OpenAI HTTP serving overhead. I added a server harness, but kernel ptrace restrictions prevented clean `py-spy` attach after server warmup. Profiling the full server path should be done either with relaxed ptrace permissions or with in-process instrumentation.
- `enforce_eager=True` disables torch.compile/CUDA graphs, so launch overhead is intentionally visible. With CUDA graphs enabled, host launch behavior can change materially.
- The model is Qwen3.5-MoE with GDN/linear attention and compressed-tensors W4A16. Function mix and kernel counts will differ for dense Transformer models.
