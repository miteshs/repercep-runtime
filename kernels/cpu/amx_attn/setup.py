"""Build script for the Repercep AMX BF16 flash-attention C++ extension.

Sits next to the GPU Triton kernels and is built on demand by the wrapper in
``src/repercep/attention/amx_flash.py``.

Invoke as::

    python setup.py build_ext --inplace

Hard requirements:
    * Sapphire Rapids (or newer) CPU with the ``amx_bf16`` flag in /proc/cpuinfo
    * gcc >= 12 (Ubuntu 24.04 ships gcc 13 which is fine)
    * PyTorch with the matching ABI
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


# ---------------------------------------------------------------------------
# Fail fast on CPUs that cannot run the kernel at all.
# ---------------------------------------------------------------------------
def _require_amx_bf16() -> None:
    if os.environ.get("REPERCEP_AMX_FORCE_BUILD") == "1":
        print(
            "[amx-bf16] REPERCEP_AMX_FORCE_BUILD=1 — bypassing CPUID gate "
            "(compile-only smoke).",
            file=sys.stderr,
        )
        return

    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        # Non-Linux build host - allow it through; the runtime check in
        # the Python wrapper will catch it.
        print("[amx_attn] /proc/cpuinfo not found, skipping CPU feature check.")
        return

    flags_found: set[str] = set()
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("flags"):
            _, _, raw = line.partition(":")
            flags_found.update(raw.split())
            break

    if "amx_bf16" not in flags_found:
        sys.stderr.write(
            "[amx_attn] FATAL: this CPU does not expose 'amx_bf16' in "
            "/proc/cpuinfo. This kernel requires Intel Sapphire Rapids or "
            "newer. Refusing to build.\n"
        )
        sys.exit(1)


_require_amx_bf16()


# ---------------------------------------------------------------------------
# Compiler flags.  -march=sapphirerapids implies most of what we need, but we
# enumerate the AMX/AVX-512 sub-features explicitly so that a slightly older
# toolchain that doesn't yet group them under -march still picks them up.
# ---------------------------------------------------------------------------
EXTRA_COMPILE_ARGS = [
    "-std=c++17",
    "-O3",
    "-fopenmp",
    "-march=sapphirerapids",
    "-mamx-bf16",
    "-mamx-tile",
    "-mamx-int8",
    "-mavx512bf16",
    "-mavx512fp16",
    "-mavx512f",
    "-mavx512vl",
    "-mavx512dq",
    "-mavx512bw",
    "-funroll-loops",
    "-fno-strict-aliasing",
    "-Wno-unused-function",
    "-Wno-unused-variable",
]

EXTRA_LINK_ARGS = ["-fopenmp"]


setup(
    name="repercep_amx_attn",
    version="0.1.0",
    description="Intel AMX BF16 flash-attention kernel for Repercep (CPU sibling "
                "of the GPU Triton kernels).",
    ext_modules=[
        CppExtension(
            name="_native",
            sources=["flash_attn_amx.cpp"],
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=EXTRA_LINK_ARGS,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
