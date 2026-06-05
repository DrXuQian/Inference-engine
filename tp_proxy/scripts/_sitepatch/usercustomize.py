"""Auto-apply the NVTX batch-size patch in EVERY Python interpreter.

Python's ``site`` module imports ``usercustomize`` at interpreter startup. When
this directory is on PYTHONPATH, that happens for the main process AND for any
process vLLM spawns (engine core / TP workers), because ``multiprocessing``
spawn re-runs site initialisation. This is the only place a monkeypatch can be
installed into the spawned worker where ``GPUModelRunner.execute_model`` (the
emitter of the ``decode bs=N`` NVTX marker) actually runs.

Activated by setting PATCH_NVTX to the path of patch_vllm_batch_nvtx.py.
The patch is applied lazily, the first time vLLM imports its model-runner
module, so vLLM's own import ordering is preserved. Everything is wrapped so a
failure here can never crash the worker.
"""

import os


def _install():
    patch_path = os.environ.get("PATCH_NVTX", "")
    if not patch_path or not os.path.exists(patch_path):
        return

    import builtins
    import importlib.util
    import sys

    _orig_import = builtins.__import__
    done = {"v": False}
    busy = {"v": False}

    def _try_apply():
        # busy guard: applying the patch re-imports vllm modules, which would
        # re-enter this hook and recurse — skip while already in progress.
        if done["v"] or busy["v"]:
            return
        # Only act once the model-runner module is actually loaded.
        if ("vllm.v1.worker.gpu_model_runner" not in sys.modules
                and "vllm.worker.model_runner" not in sys.modules):
            return
        busy["v"] = True
        try:
            spec = importlib.util.spec_from_file_location("_patch_nvtx_auto", patch_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if mod.apply():
                done["v"] = True
                builtins.__import__ = _orig_import  # restore: no further overhead
        except Exception as e:  # never break the worker
            print(f"[usercustomize] NVTX patch apply failed: {e}", file=sys.stderr)
        finally:
            busy["v"] = False

    def _hook(name, *args, **kwargs):
        m = _orig_import(name, *args, **kwargs)
        if not done["v"] and not busy["v"] and name.startswith("vllm"):
            _try_apply()
        return m

    builtins.__import__ = _hook
    # In case the module is somehow already imported before the hook is set.
    _try_apply()


try:
    _install()
except Exception:
    pass
