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
    """Apply the NVTX batch size patch to ModelRunner.execute_model."""
    try:
        import torch
        from vllm.worker.model_runner import ModelRunner

        orig = ModelRunner.execute_model
        if hasattr(orig, '_nvtx_patched'):
            return  # already patched

        def patched_execute_model(self, *args, **kwargs):
            model_input = args[0] if args else kwargs.get('model_input')
            bs = 0
            if hasattr(model_input, 'input_tokens'):
                bs = model_input.input_tokens.shape[0]
            elif hasattr(model_input, 'seq_lens'):
                bs = len(model_input.seq_lens)
            torch.cuda.nvtx.range_push(f"bs={bs}")
            result = orig(self, *args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return result

        patched_execute_model._nvtx_patched = True
        ModelRunner.execute_model = patched_execute_model
        print("[patch_vllm_batch_nvtx] Applied: ModelRunner.execute_model → NVTX bs=N")
        return True
    except Exception as e:
        print(f"[patch_vllm_batch_nvtx] Failed: {e}")
        return False


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--apply" in sys.argv:
        apply()
    else:
        print(__doc__)
