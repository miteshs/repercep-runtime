"""Compute backends for Repercep.

A backend is the policy layer above PyTorch: it declares hardware capabilities,
owns device selection, and chooses kernels appropriate to the architecture.
The Runtime depends only on the ``Backend`` Protocol, never on a concrete
vendor backend, so a second hardware target is an added implementation rather
than a rewrite (ADR-0001, ADR-0003).
"""

from __future__ import annotations

from repercep.backend.cuda import CUDABackend
from repercep.backend.protocol import Backend
from repercep.backend.registry import available_backends, select_backend
from repercep.backend.rocm import ROCmBackend

__all__ = ["Backend", "CUDABackend", "ROCmBackend", "available_backends", "select_backend"]
