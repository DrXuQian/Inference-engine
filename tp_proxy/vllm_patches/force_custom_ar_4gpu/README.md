# Force vLLM Custom All-Reduce on 4-GPU PCIe

This patch makes real vLLM use its existing custom all-reduce path for small
4-GPU packets. It does not replace vLLM's CUDA kernel. It only bypasses the
Python topology gate that normally disables custom all-reduce for more than two
PCIe-only GPUs.

For small packets, vLLM's custom all-reduce policy resolves to the oneshot
kernel.

## Use

Run vLLM with this directory at the front of `PYTHONPATH`:

```bash
PYTHONPATH=/root/autodl-tmp/Inference-engine/tp_proxy/vllm_patches/force_custom_ar_4gpu:$PYTHONPATH \
VLLM_FORCE_CUSTOM_AR_4GPU=1 \
VLLM_FORCE_CUSTOM_AR_DEBUG=1 \
VLLM_FORCE_CUSTOM_AR_MAX_WORLD=4 \
VLLM_FORCE_CUSTOM_AR_MAX_BYTES=524288 \
VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
VLLM_ALLREDUCE_USE_FLASHINFER=0 \
VLLM_SKIP_P2P_CHECK=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python your_vllm_script.py
```

`VLLM_FORCE_CUSTOM_AR_MAX_BYTES=524288` means only tensors up to 512 KB are
eligible for custom all-reduce. Larger tensors fall back to vLLM's normal
PyNccl/NCCL path.

## Verify

The process should print:

```text
[force_custom_ar_4gpu] enabled: forcing vLLM custom AR topology gate for <=4 GPUs, small packets only
```

With `VLLM_FORCE_CUSTOM_AR_DEBUG=1`, the patch also prints low-frequency
runtime checks:

```text
[force_custom_ar_4gpu] CustomAllreduce init: rank=0 world=4 disabled=False max_size=524288 fully_connected=True
[force_custom_ar_4gpu] should_custom_ar=True: rank=0 world=4 bytes=6144 dtype=torch.bfloat16 shape=(...)
[force_custom_ar_4gpu] custom_all_reduce used: rank=0 world=4 bytes=6144 dtype=torch.bfloat16 shape=(...)
```

Interpretation:

- `enabled` means the patch was imported.
- `disabled=False` means vLLM's custom all-reduce communicator initialized.
- `should_custom_ar=True` means a tensor passed vLLM's size/contiguity checks.
- `custom_all_reduce used` means this process actually entered vLLM's custom
  all-reduce path. For small packets, vLLM resolves this path to oneshot.

Check vLLM logs. If this still appears, the patch did not reach the worker
processes:

```text
Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs
```

For kernel-level confirmation, profile with `nsys` and inspect the all-reduce
kernel names:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  -o /tmp/vllm_force_custom_ar_tp4 \
  env PYTHONPATH=/root/autodl-tmp/Inference-engine/tp_proxy/vllm_patches/force_custom_ar_4gpu:$PYTHONPATH \
      VLLM_FORCE_CUSTOM_AR_4GPU=1 \
      VLLM_FORCE_CUSTOM_AR_DEBUG=1 \
      VLLM_FORCE_CUSTOM_AR_MAX_WORLD=4 \
      VLLM_FORCE_CUSTOM_AR_MAX_BYTES=524288 \
      VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
      VLLM_ALLREDUCE_USE_FLASHINFER=0 \
      VLLM_SKIP_P2P_CHECK=0 \
      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      python your_vllm_script.py
```

## Safety Notes

- This patch assumes 4-GPU P2P access works. Start with
  `VLLM_SKIP_P2P_CHECK=0` so vLLM performs a real P2P check.
- The physical topology gate is bypassed only during
  `CustomAllreduce.__init__`; vLLM's size/contiguity checks still apply.
- Keep `VLLM_FORCE_CUSTOM_AR_MAX_BYTES` small until the target platform has
  been profiled. For the current small-packet use case, 512 KB is the intended
  ceiling.
