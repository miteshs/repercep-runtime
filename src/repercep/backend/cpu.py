"""The Intel CPU backend — Sapphire Rapids AMX-first.

Repercep's third compute backend, parallel to ROCm and CUDA.  The motivation
isn't "world models on CPU at production latency" (Cosmos-7B on CPU is
minutes-per-video at best); it is closing the substrate matrix so the
Backend Protocol's promise of "one class per vendor, no model-side code
changes" carries through to the CPU floor too.  That gives us a
deterministic numerical-parity oracle (against the GPU paths), a CI fast
path that doesn't need a GPU, and — on AMX-class silicon — a competitive
*edge* path for the smaller world models on the roadmap.

Architecture mapping.  PyTorch on CPU exposes a single ``torch.device("cpu")``;
the silicon-level distinguishing signals (AMX vs plain AVX-512 vs older AVX2)
have to come from ``/proc/cpuinfo`` flags.  This module reads those flags
directly rather than spawning ``lscpu`` so the detection works inside
containers that lack the binary.

The four uarchs the registry knows about:

* **Sapphire Rapids** (``spr``) — first Intel AMX silicon (Q1-2023).
  AMX_BF16 + AMX_INT8 + AVX-512 BF16/FP16.  The host CPU on the AMD MI300X
  RunPod template and on the H100 SXM5 template both report ``spr``.
* **Emerald Rapids** (``emr``) — same AMX ISA as SPR, more cores per socket.
* **Granite Rapids** (``gnr``) — adds AMX_FP16 + AMX_COMPLEX (not yet wired).
* **Generic AVX-512** — pre-AMX Xeon (Ice Lake, Cascade Lake).  Repercep will
  run; the AMX attention ops disqualify themselves and selection falls
  through to the SDPA->oneDNN path or the naive floor.

Capabilities advertise BF16 + INT8 + FP16 when AMX is present; FP8 is not
advertised because Sapphire Rapids has no FP8 ISA support (the next Xeon
generation that gets FP8 is not yet public).  ``REPERCEP_AMX_ATTENTION``
mirrors ``REPERCEP_FP8_ATTENTION`` on the GPU side: with ``=1``, the
registry routes attention through the AMX-aware op when shape qualifies;
unset, the BF16 SDPA->oneDNN path is the floor.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from repercep.attention.registry import select_attention_op
from repercep.hardware import (
    EMERALD_RAPIDS,
    GRANITE_RAPIDS,
    SAPPHIRE_RAPIDS,
    BackendCapabilities,
    DeviceArch,
    DeviceSpec,
    DType,
    Vendor,
)

if TYPE_CHECKING:
    import torch

    from repercep.attention.protocol import AttentionOp
    from repercep.attention.types import AttentionShape


_CPUINFO_PATH = Path("/proc/cpuinfo")

# Family + model numbers for the Intel AMX-class uarchs.  Sourced from
# Intel's CPUID documentation (vol. 4, table 2-1) and cross-checked against
# the cpuid(1) tables.  Family is always 6 for modern Xeon.
#
#   Sapphire Rapids: family=6, model=143 (0x8F)
#   Emerald Rapids:  family=6, model=207 (0xCF)
#   Granite Rapids:  family=6, model=173 (0xAD)
_INTEL_MODEL_TO_ARCH: dict[tuple[int, int], DeviceArch] = {
    (6, 143): SAPPHIRE_RAPIDS,
    (6, 207): EMERALD_RAPIDS,
    (6, 173): GRANITE_RAPIDS,
}

_GENERIC_AVX512 = DeviceArch(Vendor.INTEL, "avx512", "Generic AVX-512")
_GENERIC_X86 = DeviceArch(Vendor.INTEL, "x86_64", "Generic x86-64")


@lru_cache(maxsize=1)
def _read_cpuinfo() -> dict[str, str]:
    """Parse the first processor block out of /proc/cpuinfo."""
    fields: dict[str, str] = {}
    if not _CPUINFO_PATH.exists():
        return fields
    for raw_line in _CPUINFO_PATH.read_text().splitlines():
        if not raw_line.strip():
            break
        if ":" not in raw_line:
            continue
        key, _, value = raw_line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _detect_arch_and_name() -> tuple[DeviceArch, str]:
    info = _read_cpuinfo()
    vendor_id = info.get("vendor_id", "")
    name = info.get("model name", "unknown CPU")
    if vendor_id != "GenuineIntel":
        return _GENERIC_X86, name
    try:
        family = int(info.get("cpu family", "0"))
        model = int(info.get("model", "0"))
    except ValueError:
        return _GENERIC_X86, name
    arch = _INTEL_MODEL_TO_ARCH.get((family, model))
    if arch is not None:
        return arch, name
    flags = info.get("flags", "").split()
    if "avx512f" in flags:
        return _GENERIC_AVX512, name
    return _GENERIC_X86, name


def _detect_dtypes_and_features() -> tuple[frozenset[DType], dict[str, bool]]:
    info = _read_cpuinfo()
    flags = set(info.get("flags", "").split())
    features = {
        "avx512f": "avx512f" in flags,
        "avx512_bf16": "avx512_bf16" in flags,
        "avx512_fp16": "avx512_fp16" in flags,
        "amx_tile": "amx_tile" in flags,
        "amx_bf16": "amx_bf16" in flags,
        "amx_int8": "amx_int8" in flags,
        # Granite Rapids+; not present on SPR/EMR.
        "amx_fp16": "amx_fp16" in flags,
        "avx_vnni": "avx_vnni" in flags,
    }
    dtypes: set[DType] = {DType.FP32, DType.FP16, DType.BF16, DType.INT8}
    return frozenset(dtypes), features


def _affinity_cpu_count() -> int | None:
    """``os.sched_getaffinity`` is Linux-only (absent from the darwin/win32
    stdlib stubs, not just unavailable at runtime).

    A ``sys.platform == "linux"`` guard looks like the obvious fix, but
    mypy statically evaluates that comparison against its own ``platform``
    setting (defaulting to whatever host mypy runs on) and — with
    ``warn_unreachable = true`` (set repo-wide) — flags whichever side is
    dead as unreachable. That side flips depending on which OS runs mypy
    (darwin locally, linux in CI), so it passed here and failed in CI even
    though nothing platform-relevant changed. ``getattr`` sidesteps mypy's
    platform-narrowing entirely: no literal comparison, so no branch is
    ever statically eliminated on any host.
    """
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is None:
        return None
    return len(sched_getaffinity(0))


def _detect_topology() -> tuple[int, int]:
    """Return (physical cores, logical cores) — used to size OMP_NUM_THREADS."""
    info = _read_cpuinfo()
    siblings = int(info.get("siblings", "0") or 0)
    cores = int(info.get("cpu cores", "0") or 0)
    if cores <= 0 or siblings <= 0:
        logical = _affinity_cpu_count() or os.cpu_count() or 1
        return max(logical // 2, 1), logical
    logical = _affinity_cpu_count() or os.cpu_count() or siblings
    sockets = max(logical // siblings, 1)
    return cores * sockets, logical


def _system_ram_bytes() -> int:
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return 0
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            try:
                return int(parts[1]) * 1024
            except (IndexError, ValueError):
                return 0
    return 0


class CPUBackend:
    """Intel-CPU backend.  Lead target: Sapphire Rapids (AMX_BF16 + AMX_INT8)."""

    vendor: Vendor = Vendor.INTEL
    name: str = "cpu"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        # CPU is always available where torch is.  The backend registry
        # ordering (ROCm -> CUDA -> CPU) makes this the floor, not the
        # default on a GPU host.
        return True

    def devices(self) -> tuple[DeviceSpec, ...]:
        arch, name = _detect_arch_and_name()
        _, logical = _detect_topology()
        return (
            DeviceSpec(
                index=0,
                arch=arch,
                name=name,
                total_memory_bytes=_system_ram_bytes(),
                multi_processor_count=logical,
            ),
        )

    def capabilities(self) -> BackendCapabilities:
        from repercep.attention.amx_sdpa import AMXSDPAAttention

        dtypes, features = _detect_dtypes_and_features()
        ops: list[str] = []

        if features["amx_bf16"]:
            try:
                from repercep.attention.amx_flash import AMXFlashAttention

                if AMXFlashAttention().available:
                    ops.append("amx-bf16-flash")
            except ImportError:
                pass
            try:
                from repercep.attention.ipex_flash import IPEXFlashAttention

                if IPEXFlashAttention().available:
                    ops.append("ipex-flash")
            except ImportError:
                pass
        if features["amx_int8"]:
            try:
                from repercep.attention.amx_int8_flash import AMXInt8FlashAttention

                if AMXInt8FlashAttention().available:
                    ops.append("amx-int8-flash")
            except ImportError:
                pass
        if features["amx_fp16"]:
            # Granite Rapids+ only.  SPR/EMR report amx_bf16 + amx_int8 but
            # not amx_fp16; the FP16 op disqualifies cleanly on those.
            try:
                from repercep.attention.amx_fp16_flash import AMXFP16FlashAttention

                if AMXFP16FlashAttention().available:
                    ops.append("amx-fp16-flash")
            except ImportError:
                pass
        sdpa = AMXSDPAAttention()
        if sdpa.available:
            ops.append(sdpa.name)
        ops.append("naive-sdpa")

        supports_flash = any(
            o in ops
            for o in ("amx-bf16-flash", "amx-int8-flash", "amx-fp16-flash", "ipex-flash")
        )
        return BackendCapabilities(
            dtypes=dtypes,
            supports_flash_attention=supports_flash,
            supports_fp8=False,
            supports_torch_compile=True,
            attention_ops=tuple(ops),
        )

    def torch_device(self, index: int = 0) -> torch.device:
        import torch

        return torch.device("cpu")

    def default_dtype(self) -> DType:
        # BF16 is the AMX speed lever.  On a non-AMX CPU this falls back to
        # the slow ATen BF16 path, but BF16 is still the right default since
        # weight footprint matters more than per-op throughput on a
        # memory-bound host.
        return DType.BF16

    def attention_op(self, shape: AttentionShape, dtype: DType) -> AttentionOp:
        devices = self.devices()
        arch = SAPPHIRE_RAPIDS if not devices else devices[0].arch
        return select_attention_op(arch, shape, dtype)
