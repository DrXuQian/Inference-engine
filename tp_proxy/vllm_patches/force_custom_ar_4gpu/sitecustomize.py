"""Force vLLM custom all-reduce for small 4-GPU PCIe packets.

Put this directory at the front of PYTHONPATH before starting vLLM:

  PYTHONPATH=/path/to/force_custom_ar_4gpu:$PYTHONPATH \
  VLLM_FORCE_CUSTOM_AR_4GPU=1 \
  VLLM_FORCE_CUSTOM_AR_MAX_BYTES=524288 \
  python your_vllm_script.py

The patch only changes vLLM's Python topology gate. It does not modify vLLM's
CUDA kernels. The actual small-packet kernel is still vLLM's custom all-reduce
oneshot path.
"""

from __future__ import annotations

import os


def _enabled() -> bool:
    return os.getenv("VLLM_FORCE_CUSTOM_AR_4GPU", "0") == "1"


def _install_patch() -> None:
    try:
        from vllm.distributed.device_communicators import custom_all_reduce as car
        from vllm.platforms import current_platform
    except Exception as exc:  # pragma: no cover - only hit outside vLLM envs.
        print(f"[force_custom_ar_4gpu] skipped: cannot import vLLM ({exc})", flush=True)
        return

    if getattr(car.CustomAllreduce, "_force_4gpu_patch_installed", False):
        return

    orig_init = car.CustomAllreduce.__init__

    def patched_init(self, group, device, max_size=8192 * 1024, symm_mem_enabled=False):
        # Keep the override scoped to small packets. For larger tensors,
        # should_custom_ar() returns False and vLLM falls back to PyNccl/NCCL.
        force_max = int(os.getenv("VLLM_FORCE_CUSTOM_AR_MAX_BYTES", str(512 * 1024)))
        max_size = min(max_size, force_max)

        max_world = int(os.getenv("VLLM_FORCE_CUSTOM_AR_MAX_WORLD", "4"))
        orig_is_fully_connected = current_platform.is_fully_connected

        def forced_is_fully_connected(device_ids):
            if len(device_ids) <= max_world:
                return True
            return orig_is_fully_connected(device_ids)

        current_platform.is_fully_connected = forced_is_fully_connected
        try:
            return orig_init(
                self,
                group,
                device,
                max_size=max_size,
                symm_mem_enabled=symm_mem_enabled,
            )
        finally:
            current_platform.is_fully_connected = orig_is_fully_connected

    car.CustomAllreduce.__init__ = patched_init
    car.CustomAllreduce._force_4gpu_patch_installed = True
    print(
        "[force_custom_ar_4gpu] enabled: forcing vLLM custom AR topology gate "
        "for <=4 GPUs, small packets only",
        flush=True,
    )


if _enabled():
    _install_patch()
