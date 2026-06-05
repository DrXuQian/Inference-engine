#!/usr/bin/env python3
"""
Inject NVTX prefill/decode markers directly into vllm's gpu_model_runner.py.

This is needed because vLLM v1 runs workers in subprocesses — monkeypatches
from the main process don't survive. This script modifies the installed source.

Usage:
    python inject_nvtx_to_vllm.py          # apply patch
    python inject_nvtx_to_vllm.py --check  # check if already patched
    python inject_nvtx_to_vllm.py --revert # revert patch
"""

import sys
import shutil


def find_source():
    try:
        from vllm.v1.worker import gpu_model_runner
        return gpu_model_runner.__file__
    except ImportError:
        # v0 fallback
        from vllm.worker import model_runner
        return model_runner.__file__


MARKER = "# --- NVTX_INJECT_MARKER ---"

INJECT_CODE = '''
        # --- NVTX_INJECT_MARKER ---
        import nvtx as _nvtx_mod
        _nvtx_ntoks = scheduler_output.total_num_scheduled_tokens
        _nvtx_max = max(scheduler_output.num_scheduled_tokens.values()) if scheduler_output.num_scheduled_tokens else 0
        _nvtx_phase = "prefill" if _nvtx_max > 1 else "decode"
        _nvtx_rng = _nvtx_mod.start_range(f"{_nvtx_phase} tok={_nvtx_ntoks}")
        # --- END NVTX_INJECT_MARKER ---'''

CLEANUP_CODE = '''
        # --- NVTX_CLEANUP_MARKER ---
        _nvtx_mod.end_range(_nvtx_rng)
        # --- END NVTX_CLEANUP_MARKER ---'''

CLEANUP_MARKER = "# --- NVTX_CLEANUP_MARKER ---"


def check(path):
    with open(path) as f:
        src = f.read()
    if MARKER in src:
        print(f"PATCHED: {path}")
        return True
    print(f"NOT PATCHED: {path}")
    return False


def apply(path):
    with open(path) as f:
        lines = f.readlines()

    if MARKER in "".join(lines):
        print(f"Already patched: {path}")
        return

    # Backup
    backup = path + ".bak"
    shutil.copy2(path, backup)
    print(f"Backup: {backup}")

    # Find "def execute_model" with scheduler_output parameter
    inject_line = None
    for i, line in enumerate(lines):
        if "def execute_model" in line and "scheduler_output" in line:
            # Find the end of the function signature (the line with "):")
            j = i
            while j < len(lines) and "):" not in lines[j]:
                j += 1
            inject_line = j + 1  # insert after the "def ...():" line
            break

    if inject_line is None:
        print("ERROR: could not find 'def execute_model(...scheduler_output...)' in source")
        sys.exit(1)

    # Find return statements in execute_model to add cleanup
    # We'll add cleanup before the first return that's at the method's indentation level
    # Actually, simpler: wrap the whole body in try/finally
    # But that's complex. Instead, just add start_range at entry - the range will auto-close
    # when the profiler stops, which is fine for our purpose.

    # Actually nvtx ranges that aren't closed just stay open, which is messy.
    # Let's use a simpler approach: use nvtx.annotate as a context manager...
    # but that requires wrapping the whole function body.
    #
    # Simplest correct approach: use torch's record_function or just accept unclosed ranges.
    # For profiling purposes, unclosed ranges at the very end are fine - the profiler
    # captures the start timestamp which is what we need to distinguish prefill vs decode.
    #
    # Actually, let me just inject the start_range. The range will be visible in the trace
    # even without end_range - it just won't have a duration, but the TEXT is what matters
    # for distinguishing prefill from decode.

    # Better: use nvtx.annotate as a decorator-like pattern with a helper.
    # Simplest: just do start_range, and add end_range before every return.

    # Find all returns in execute_model at the correct indentation
    # Get base indentation of the method body
    body_indent = None
    for k in range(inject_line, len(lines)):
        stripped = lines[k].lstrip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
            body_indent = len(lines[k]) - len(lines[k].lstrip())
            break

    # Insert cleanup before returns
    new_lines = lines[:inject_line]
    new_lines.append(INJECT_CODE + "\n")

    # Track brace/indent level to know when execute_model ends
    in_method = True
    for k in range(inject_line, len(lines)):
        line = lines[k]
        stripped = line.lstrip()
        curr_indent = len(line) - len(line.lstrip()) if stripped else 999

        # Check if we've left the method (dedented to class level)
        if stripped and curr_indent < body_indent and not stripped.startswith('@'):
            in_method = False

        if in_method and stripped.startswith("return "):
            # Add end_range before return
            indent = " " * curr_indent
            new_lines.append(f"{indent}_nvtx_mod.end_range(_nvtx_rng)  {CLEANUP_MARKER}\n")

        new_lines.append(line)

    with open(path, "w") as f:
        f.writelines(new_lines)

    print(f"Patched: {path}")
    print(f"  Injected NVTX markers into execute_model()")


def revert(path):
    backup = path + ".bak"
    try:
        shutil.copy2(backup, path)
        print(f"Reverted: {path} (from {backup})")
    except FileNotFoundError:
        # Try removing injected lines manually
        with open(path) as f:
            lines = f.readlines()
        new_lines = []
        skip = False
        for line in lines:
            if "NVTX_INJECT_MARKER" in line:
                skip = True
                continue
            if "END NVTX_INJECT_MARKER" in line:
                skip = False
                continue
            if CLEANUP_MARKER in line:
                continue
            if not skip:
                new_lines.append(line)
        with open(path, "w") as f:
            f.writelines(new_lines)
        print(f"Reverted: {path} (removed injected lines)")


def main():
    path = find_source()
    print(f"Source: {path}")

    if "--check" in sys.argv:
        check(path)
    elif "--revert" in sys.argv:
        revert(path)
    else:
        apply(path)


if __name__ == "__main__":
    main()
