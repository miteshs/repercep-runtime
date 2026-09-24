"""Repercep Runtime — a world-model-native inference engine.

Lead workload: Cosmos-Predict-7B.  Lead hardware: AMD Instinct MI300X (gfx942, CDNA3).

The codebase is vendor-neutral by construction (see ``repercep.backend.protocol``):
MI300X is the first and currently only concrete backend; an NVIDIA backend is a
zero-rewrite fast-follow.  See ``docs/architecture.md`` and ``docs/adr/`` for the
design rationale, including why MI300X leads (ADR-0001).
"""

from __future__ import annotations

from repercep.hardware import (
    H100,
    H200,
    MI250X,
    MI300X,
    BackendCapabilities,
    DeviceArch,
    DeviceSpec,
    DType,
    Vendor,
)

__version__ = "0.0.1"

__all__ = [
    "H100",
    "H200",
    "MI250X",
    "MI300X",
    "BackendCapabilities",
    "DType",
    "DeviceArch",
    "DeviceSpec",
    "Vendor",
    "__version__",
]
