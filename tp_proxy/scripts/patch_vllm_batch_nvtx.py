#!/usr/bin/env python3
"""
Monkeypatch vLLM ModelRunner to add NVTX batch size markers.

Usage:
    # Check if patch is needed
    python patch_vllm_batch_nvtx.py --check

    # Apply patch (import before creating LLM)
    python patch_vllm_batch_nvtx.py --apply

    # In your script:
    import patch_vllm_batch_nvtx
    patch_vllm_batch_nvtx.apply()
    from vllm import LLM  # now patched
"""

import sys


def check():
    """Check if ModelRunner can be patched."""
    try:
        from vllm.worker.model_runner import ModelRunner
        orig = ModelRunner.execute_model
        # Check if already patched
        if hasattr(orig, '_nvtx_patched'):
            print("ALREADY PATCHED")
            return True
        print("NOT PATCHED (ready to apply)")
        print(f"  ModelRunner: {ModelRunner.__module__}")
        print(f"  execute_model: {orig}")
        return True
    except ImportError as e:
        print(f"CANNOT PATCH: {e}")
        return False


def apply():
    """Apply the NVTX prefill/decode + batch size patch to ModelRunner.execute_model."""
    patched = False

    # Try vLLM v1 (0.19+): gpu_model_runner.GPUModelRunner
    try:
        import torch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        orig = GPUModelRunner.execute_model
        if hasattr(orig, '_nvtx_patched'):
            return True

        import nvtx as _nvtx

        def patched_v1(self, scheduler_output, **kwargs):
            num_tokens = scheduler_output.total_num_scheduled_tokens
            num_reqs = len(scheduler_output.num_scheduled_tokens)
            max_toks = max(scheduler_output.num_scheduled_tokens.values()) if num_reqs else 0
            phase = "prefill" if max_toks > 1 else "decode"
            rng = _nvtx.start_range(f"{phase} bs={num_reqs} tok={num_tokens}")
            result = orig(self, scheduler_output, **kwargs)
            _nvtx.end_range(rng)
            return result

        patched_v1._nvtx_patched = True
        GPUModelRunner.execute_model = patched_v1
        print("[patch_vllm_batch_nvtx] Applied v1: GPUModelRunner.execute_model "
              "→ NVTX {prefill|decode} bs=N tok=N")
        patched = True
    except (ImportError, AttributeError):
        pass

    # Fallback: vLLM v0 (legacy): worker.model_runner.ModelRunner
    if not patched:
        try:
            import torch
            from vllm.worker.model_runner import ModelRunner

            orig = ModelRunner.execute_model
            if hasattr(orig, '_nvtx_patched'):
                return True

            import nvtx as _nvtx

            def patched_v0(self, *args, **kwargs):
                model_input = args[0] if args else kwargs.get('model_input')
                bs = 0
                if hasattr(model_input, 'input_tokens'):
                    bs = model_input.input_tokens.shape[0]
                elif hasattr(model_input, 'seq_lens'):
                    bs = len(model_input.seq_lens)
                rng = _nvtx.start_range(f"bs={bs}")
                result = orig(self, *args, **kwargs)
                _nvtx.end_range(rng)
                return result

            patched_v0._nvtx_patched = True
            ModelRunner.execute_model = patched_v0
            print("[patch_vllm_batch_nvtx] Applied v0: ModelRunner.execute_model → NVTX bs=N")
            patched = True
        except (ImportError, AttributeError):
            pass

    if not patched:
        print("[patch_vllm_batch_nvtx] Failed: no compatible ModelRunner found")
    return patched


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--apply" in sys.argv:
        apply()
    else:
        print(__doc__)
