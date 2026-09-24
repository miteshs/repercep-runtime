"""Core hardware domain types for Repercep.

These types are deliberately framework-agnostic: they describe *what* a piece of
silicon is and can do, without importing torch.  The backend layer
(``repercep.backend``) maps these descriptions onto a concrete compute runtime.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Vendor(enum.StrEnum):
    """Compute vendor.  Disambiguates the ``gfx``/``sm``/CPU-uarch namespaces."""

    AMD = "amd"
    NVIDIA = "nvidia"
    INTEL = "intel"


class DType(enum.StrEnum):
    """Numeric types Repercep reasons about for inference and kernel selection."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"


@dataclass(frozen=True, slots=True)
class DeviceArch:
    """A GPU architecture target.

    ``gfx_id`` is the LLVM/ROCm target (e.g. ``gfx942`` for MI300X/MI300A) or the
    CUDA SM string (e.g. ``sm90a`` for H100).  ``vendor`` disambiguates the two
    namespaces so ``gfx942`` and a hypothetical ``sm942`` never collide.
    """

    vendor: Vendor
    gfx_id: str
    family: str  # human-readable micro-architecture, e.g. "CDNA3", "Hopper"

    def __str__(self) -> str:
        return f"{self.vendor.value}:{self.gfx_id}"


# --- Architectures Repercep reasons about -------------------------------------
# MI300X is the lead target (ADR-0001).  The NVIDIA entries exist so the
# vendor-neutral typing is real today; no NVIDIA backend is implemented yet.
MI300X = DeviceArch(Vendor.AMD, "gfx942", "CDNA3")
MI250X = DeviceArch(Vendor.AMD, "gfx90a", "CDNA2")
H100 = DeviceArch(Vendor.NVIDIA, "sm90a", "Hopper")
H200 = DeviceArch(Vendor.NVIDIA, "sm90a", "Hopper")
# Intel CPU uarchs that carry AMX (the lever for matmul on CPU).  Sapphire
# Rapids was the first AMX part (Q1-2023); Emerald Rapids inherits the same
# AMX_BF16 + AMX_INT8 ISA; Granite Rapids adds AMX_FP16 + AMX_COMPLEX.
# Older Xeon parts (Ice Lake, Cascade Lake) fall back to plain AVX-512.
SAPPHIRE_RAPIDS = DeviceArch(Vendor.INTEL, "spr", "Sapphire Rapids")
EMERALD_RAPIDS = DeviceArch(Vendor.INTEL, "emr", "Emerald Rapids")
GRANITE_RAPIDS = DeviceArch(Vendor.INTEL, "gnr", "Granite Rapids")


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """A concrete, addressable accelerator on this host."""

    index: int
    arch: DeviceArch
    name: str
    total_memory_bytes: int
    multi_processor_count: int  # AMD "compute units" / NVIDIA "SMs"

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory_bytes / (1024**3)


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """What a backend can actually do on the current host.

    ``attention_ops`` is ordered by preference (fastest first); the Runtime
    never reads it directly — it asks the backend for an op via
    ``Backend.attention_op`` — but it is surfaced for diagnostics and ``repercep
    info``.
    """

    dtypes: frozenset[DType]
    supports_flash_attention: bool
    supports_fp8: bool
    supports_torch_compile: bool
    attention_ops: tuple[str, ...]
