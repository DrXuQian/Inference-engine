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


def _debug_enabled() -> bool:
    return os.getenv("VLLM_FORCE_CUSTOM_AR_DEBUG", "0") == "1"


def _debug_attempt_limit() -> int:
    return int(os.getenv("VLLM_FORCE_CUSTOM_AR_DEBUG_ATTEMPTS", "16"))


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
    orig_should_custom_ar = car.CustomAllreduce.should_custom_ar
    orig_custom_all_reduce = car.CustomAllreduce.custom_all_reduce

    def reject_reason(self, inp) -> str:
        if getattr(self, "disabled", True):
            return "disabled"

        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return "bytes_not_multiple_of_16"

        if not car.is_weak_contiguous(inp):
            return "not_weak_contiguous"

        world_size = getattr(self, "world_size", None)
        fully_connected = getattr(self, "fully_connected", False)
        if not (world_size == 2 or fully_connected):
            return f"topology world={world_size} fully_connected={fully_connected}"

        max_size = getattr(self, "max_size", None)
        if max_size is not None and inp_size >= max_size:
            return f"size_ge_max_size {inp_size}>={max_size}"

        return "unknown"

    def log_should_decision(self, inp, ok: bool) -> None:
        if not _debug_enabled():
            return

        limit = _debug_attempt_limit()
        if limit <= 0:
            return

        inp_size = inp.numel() * inp.element_size()
        reason = "ok" if ok else reject_reason(self, inp)
        key = (
            ok,
            reason,
            inp_size,
            str(inp.dtype),
            tuple(inp.shape),
        )
        seen = getattr(self, "_force_4gpu_should_seen", set())
        count = getattr(self, "_force_4gpu_should_log_count", 0)
        if key in seen or count >= limit:
            return

        seen.add(key)
        self._force_4gpu_should_seen = seen
        self._force_4gpu_should_log_count = count + 1
        print(
            f"[force_custom_ar_4gpu] should_custom_ar={ok}: "
            f"rank={getattr(self, 'rank', '?')} "
            f"world={getattr(self, 'world_size', '?')} "
            f"bytes={inp_size} "
            f"max_size={getattr(self, 'max_size', '?')} "
            f"reason={reason} "
            f"dtype={inp.dtype} shape={tuple(inp.shape)}",
            flush=True,
        )

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
            ret = orig_init(
                self,
                group,
                device,
                max_size=max_size,
                symm_mem_enabled=symm_mem_enabled,
            )
            if _debug_enabled():
                print(
                    "[force_custom_ar_4gpu] CustomAllreduce init: "
                    f"rank={getattr(self, 'rank', '?')} "
                    f"world={getattr(self, 'world_size', '?')} "
                    f"disabled={getattr(self, 'disabled', '?')} "
                    f"max_size={getattr(self, 'max_size', max_size)} "
                    f"fully_connected={getattr(self, 'fully_connected', '?')}",
                    flush=True,
                )
            return ret
        finally:
            current_platform.is_fully_connected = orig_is_fully_connected

    def patched_should_custom_ar(self, inp):
        ok = orig_should_custom_ar(self, inp)
        log_should_decision(self, inp, ok)
        return ok

    def patched_custom_all_reduce(self, input):
        out = orig_custom_all_reduce(self, input)
        if (
            out is not None
            and _debug_enabled()
            and not getattr(self, "_force_4gpu_used_logged", False)
        ):
            self._force_4gpu_used_logged = True
            print(
                "[force_custom_ar_4gpu] custom_all_reduce used: "
                f"rank={getattr(self, 'rank', '?')} "
                f"world={getattr(self, 'world_size', '?')} "
                f"bytes={input.numel() * input.element_size()} "
                f"dtype={input.dtype} shape={tuple(input.shape)}",
                flush=True,
            )
        return out

    car.CustomAllreduce.__init__ = patched_init
    car.CustomAllreduce.should_custom_ar = patched_should_custom_ar
    car.CustomAllreduce.custom_all_reduce = patched_custom_all_reduce
    car.CustomAllreduce._force_4gpu_patch_installed = True
    print(
        "[force_custom_ar_4gpu] enabled: forcing vLLM custom AR topology gate "
        "for <=4 GPUs, small packets only",
        flush=True,
    )


if _enabled():
    _install_patch()
