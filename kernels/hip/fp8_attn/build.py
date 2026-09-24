"""JIT-load the HIP FP8 GEMM proof-of-life kernel.

This is the lightweight alternative to the CMakeLists.txt build — it uses
``torch.utils.cpp_extension.load`` to compile the kernel on first import.
Suitable for development; CMake is the production path (see kernels/README.md
"Future layout" — build.cmake is the future single entry point).

Usage:
    from kernels.hip.fp8_attn.build import load_hip_fp8
    ext = load_hip_fp8()
    C = ext.fp8_gemm(A_fp8, B_fp8, a_scale, b_scale)

Status: this is the kernel-layer toolchain smoke test.  The perf-path FP8
attention path lives in kernels/triton_kernels/fp8_flash_attn.py and is
already integrated as src/repercep/attention/fp8_triton.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_KERNEL_DIR = Path(__file__).resolve().parent
_EXT: Any | None = None


def load_hip_fp8(verbose: bool = False) -> Any:
    """Compile and load the HIP FP8 GEMM extension.

    Returns the loaded module.  Subsequent calls return the cached handle.
    """
    global _EXT
    if _EXT is not None:
        return _EXT

    from torch.utils.cpp_extension import load

    _EXT = load(
        name="repercep_fp8_hip",
        sources=[
            str(_KERNEL_DIR / "binding.cpp"),
            str(_KERNEL_DIR / "fp8_gemm.hip"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--offload-arch=gfx942",
            "-ffast-math",
        ],
        verbose=verbose,
    )
    return _EXT
