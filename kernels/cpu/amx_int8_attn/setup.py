"""Build script for the Repercep AMX INT8 flash-attention C++ extension.

Sibling of the BF16 build script in ``kernels/cpu/amx_attn/setup.py``.  Built
on demand by the wrapper in ``src/repercep/attention/amx_int8_flash.py``.

Invoke as::

    python setup.py build_ext --inplace

Hard requirements:
    * Sapphire Rapids (or newer) CPU with the ``amx_int8`` flag in /proc/cpuinfo
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
#
# Parameterised twin of ``_require_amx_bf16`` in ``../amx_attn/setup.py``.
# ---------------------------------------------------------------------------
def _require_cpu_flag(flag: str, kernel_label: str = "amx_int8_attn") -> None:
    if os.environ.get("REPERCEP_AMX_FORCE_BUILD") == "1":
        print(
            "[amx-int8] REPERCEP_AMX_FORCE_BUILD=1 — bypassing CPUID gate "
            "(compile-only smoke).",
            file=sys.stderr,
        )
        return

    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        # Non-Linux build host - allow it through; the runtime check in
        # the Python wrapper will catch it.
        print(f"[{kernel_label}] /proc/cpuinfo not found, skipping CPU feature check.")
        return

    flags_found: set[str] = set()
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("flags"):
            _, _, raw = line.partition(":")
            flags_found.update(raw.split())
            break

    if flag not in flags_found:
        sys.stderr.write(
            f"[{kernel_label}] FATAL: this CPU does not expose {flag!r} in "
            "/proc/cpuinfo. This kernel requires Intel Sapphire Rapids or "
            "newer. Refusing to build.\n"
        )
        sys.exit(1)


_require_cpu_flag("amx_int8")


# ---------------------------------------------------------------------------
# Compiler flags.  -march=sapphirerapids implies most of what we need; we
# enumerate the AMX/AVX-512 sub-features explicitly so that a slightly older
# toolchain that doesn't yet group them under -march still picks them up.
# Mirrors the BF16 sibling's flag list verbatim (both kernels need the same
# baseline AVX-512 + BF16-cast + AMX-tile state).
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
    name="repercep_amx_int8_attn",
    version="0.1.0",
    description="Intel AMX INT8 flash-attention kernel for Repercep "
                "(TDPBSSD-based CPU sibling of the GPU FP8 Triton kernels).",
    ext_modules=[
        CppExtension(
            name="_native",
            sources=["flash_attn_amx_int8.cpp"],
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=EXTRA_LINK_ARGS,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
