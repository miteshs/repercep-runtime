"""Build script for the Repercep AMX FP16 flash-attention C++ extension.

Granite Rapids (GNR) sibling of `kernels/cpu/amx_attn/setup.py`.  GNR is the
first Intel silicon that exposes `AMX_FP16` (`_tile_dpfp16ps`) -- SPR and
EMR do not, so this kernel cannot even compile against the matching
``-march`` target on those hosts.

Invoke as::

    python setup.py build_ext --inplace

Hard requirements:
    * Intel **Granite Rapids** CPU with the ``amx_fp16`` flag in
      ``/proc/cpuinfo`` (expected H2 2026).
    * gcc >= 14 for ``-march=graniterapids``; older toolchains can target
      the sub-feature directly via ``-mamx-fp16`` (gcc 13 + binutils 2.41).
    * PyTorch with the matching ABI.
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
# We gate on the **GNR-specific** `amx_fp16` flag rather than the SPR/EMR
# `amx_bf16` flag -- the BF16 sibling already covers SPR/EMR; this build is
# only valid on hardware that ships TDPFP16PS.
# ---------------------------------------------------------------------------
def _require_amx_fp16() -> None:
    if os.environ.get("REPERCEP_AMX_FORCE_BUILD") == "1":
        print(
            "[amx-fp16] REPERCEP_AMX_FORCE_BUILD=1 — bypassing CPUID gate "
            "(compile-only smoke).",
            file=sys.stderr,
        )
        return

    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        # Non-Linux build host - allow it through; the runtime check in
        # the Python wrapper will catch it.
        print("[amx_fp16_attn] /proc/cpuinfo not found, skipping CPU feature check.")
        return

    flags_found: set[str] = set()
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("flags"):
            _, _, raw = line.partition(":")
            flags_found.update(raw.split())
            break

    if "amx_fp16" not in flags_found:
        sys.stderr.write(
            "[amx_fp16_attn] FATAL: this CPU does not expose 'amx_fp16' in "
            "/proc/cpuinfo. This kernel requires Intel Granite Rapids or "
            "newer (SPR/EMR carry only AMX_BF16 -- use the amx_attn sibling "
            "instead). Refusing to build.\n"
        )
        sys.exit(1)


_require_amx_fp16()


# ---------------------------------------------------------------------------
# Compiler flags.  `-march=graniterapids` is the canonical target on gcc 14+;
# older gcc accepts the individual AMX_FP16 sub-feature via `-mamx-fp16`,
# so we pass both and let the toolchain pick the one it knows.  AVX-512_FP16
# (full ISA, not just the AMX tile op) is also on GNR baseline.
# ---------------------------------------------------------------------------
EXTRA_COMPILE_ARGS = [
    "-std=c++17",
    "-O3",
    "-fopenmp",
    "-march=graniterapids",
    "-mamx-fp16",
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
    name="repercep_amx_fp16_attn",
    version="0.1.0",
    description="Intel AMX FP16 flash-attention kernel for Repercep "
                "(Granite Rapids sibling of amx_attn).",
    ext_modules=[
        CppExtension(
            name="_native",
            sources=["flash_attn_amx_fp16.cpp"],
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=EXTRA_LINK_ARGS,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
