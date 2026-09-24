"""Backend discovery and selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from repercep.backend.cpu import CPUBackend
from repercep.backend.cuda import CUDABackend
from repercep.backend.rocm import ROCmBackend

if TYPE_CHECKING:
    from repercep.backend.protocol import Backend

# Every backend Repercep knows how to construct.  AMD remains the lead workload
# (ADR-0001); NVIDIA lands as a parallel track per ADR-0006; CPU/AMX is the
# substrate floor per ADR-0007.  Order matters when multiple backends are
# present on the same host — ``select_backend()`` returns the first
# available, and a GPU should always preempt CPU on a GPU host.  ROCm-first
# preserves the "MI300X is lead" framing; CUDA-second covers NVIDIA hosts;
# CPU-last is the always-available substrate that never wins selection on
# a GPU host but enables CI without a GPU.
_ALL_BACKENDS: tuple[Backend, ...] = (ROCmBackend(), CUDABackend(), CPUBackend())


def available_backends() -> tuple[Backend, ...]:
    """Backends whose hardware is actually present on this host."""
    return tuple(b for b in _ALL_BACKENDS if b.is_available())


def select_backend(prefer: str | None = None) -> Backend:
    """Return a usable backend.

    Args:
        prefer: pin a specific backend by name (e.g. ``"rocm"``, ``"cuda"``,
            ``"cpu"``).  If ``None``, the first available backend is returned
            in declaration order (GPU vendors before CPU).

    Raises:
        ValueError: ``prefer`` names a backend Repercep does not know.
        RuntimeError: the requested (or any) backend is not available.
    """
    if prefer is not None:
        for backend in _ALL_BACKENDS:
            if backend.name == prefer:
                if not backend.is_available():
                    raise RuntimeError(f"backend {prefer!r} is not available on this host")
                return backend
        known = ", ".join(b.name for b in _ALL_BACKENDS)
        raise ValueError(f"unknown backend {prefer!r}; known backends: {known}")

    for backend in _ALL_BACKENDS:
        if backend.is_available():
            return backend
    raise RuntimeError(
        "no Repercep backend is available — is torch installed? "
        "Run `python scripts/check_gpu.py` to diagnose."
    )
