#!/usr/bin/env python3
"""Standalone GPU smoke test — works on ROCm (MI300X) or CUDA (H100/H200).

Run with the project venv:  .venv/bin/python scripts/check_gpu.py

Intentionally has no Repercep imports so it works before the package is
installed — it is the first thing to run on a fresh box.  Auto-detects
which vendor's stack torch is built against and reports accordingly:

- ROCm (``torch.version.hip`` populated): MI300X / MI250X paths.
- CUDA (``torch.version.cuda`` populated, ``hip`` empty): H100 / H200 /
  Ada / Ampere paths.

A small matmul micro-benchmark at the end is a basic sanity check
(verifies both the wheel and the device are healthy) and does not
attempt to reach published TFLOP/s — for that, see
``scripts/profile_cosmos.py``.
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed in this environment.", file=sys.stderr)
        return 1

    print(f"torch   : {torch.__version__}")
    print(f"hip     : {torch.version.hip}")
    print(f"cuda    : {torch.version.cuda}")

    is_rocm = bool(torch.version.hip)
    is_cuda = torch.version.cuda is not None and not torch.version.hip

    if not is_rocm and not is_cuda:
        print(
            "FAIL: this torch build is neither a ROCm nor a CUDA build.",
            file=sys.stderr,
        )
        return 1
    if not torch.cuda.is_available():
        if is_rocm:
            print(
                "FAIL: no GPU visible (ROCm wheel). Check that /dev/kfd and "
                "/dev/dri/renderD* are readable by this user (group 'render').",
                file=sys.stderr,
            )
        else:
            print(
                "FAIL: no GPU visible (CUDA wheel). Check that /dev/nvidia* are "
                "readable by this user and that nvidia-smi succeeds.",
                file=sys.stderr,
            )
        return 1

    vendor = "ROCm/AMD" if is_rocm else "CUDA/NVIDIA"
    print(f"vendor  : {vendor}")

    count = torch.cuda.device_count()
    print(f"devices : {count}")
    for i in range(count):
        p = torch.cuda.get_device_properties(i)
        # ROCm populates ``gcnArchName`` with the LLVM target (e.g. "gfx942");
        # CUDA's properties object has the attribute too but echoes
        # ``p.name`` into it (e.g. "NVIDIA H100 80GB HBM3").  Distinguish by
        # vendor, not by attribute presence.
        arch_id = p.gcnArchName.split(":")[0] if is_rocm else f"sm_{p.major}{p.minor}"
        print(
            f"  [{i}] {p.name}  {arch_id}  "
            f"{p.total_memory / 1024**3:.0f} GiB  {p.multi_processor_count} "
            f"{'CUs' if is_rocm else 'SMs'}"
        )

    dev = torch.device("cuda", 0)
    n = 4096
    for dtype in (torch.float16, torch.bfloat16):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        for _ in range(3):  # warmup
            _ = a @ b
        torch.cuda.synchronize()
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        tflops = 2 * n**3 / (ms * 1e-3) / 1e12
        print(f"  matmul {n}x{n} {dtype!s:>16}: {ms:7.3f} ms  ({tflops:6.1f} TFLOP/s)")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
