"""Tests for the backend layer. GPU-dependent tests skip cleanly without one."""

from __future__ import annotations

import pytest

from repercep.backend.protocol import Backend
from repercep.backend.registry import select_backend
from repercep.backend.rocm import ROCmBackend
from repercep.hardware import Vendor


def test_rocm_backend_identity() -> None:
    backend = ROCmBackend()
    assert backend.vendor is Vendor.AMD
    assert backend.name == "rocm"


def test_rocm_backend_satisfies_protocol() -> None:
    # The whole point of the Protocol: structural conformance, checked here.
    assert isinstance(ROCmBackend(), Backend)


def test_select_unknown_backend_raises() -> None:
    # "tpu" is not a backend Repercep knows about today; CUDA was added in ADR-0006
    # so it no longer makes a good "unknown" example.
    with pytest.raises(ValueError, match="unknown backend"):
        select_backend(prefer="tpu")


@pytest.mark.skipif(not ROCmBackend().is_available(), reason="no ROCm GPU on host")
def test_rocm_devices_detected() -> None:
    backend = ROCmBackend()
    devices = backend.devices()
    assert len(devices) >= 1
    assert devices[0].arch.gfx_id.startswith("gfx")
    assert devices[0].total_memory_bytes > 0


@pytest.mark.skipif(not ROCmBackend().is_available(), reason="no ROCm GPU on host")
def test_select_backend_returns_rocm() -> None:
    assert select_backend().name == "rocm"


@pytest.mark.skipif(not ROCmBackend().is_available(), reason="no ROCm GPU on host")
def test_capabilities_report_fp8() -> None:
    # CDNA3 has native FP8 MFMA.
    assert ROCmBackend().capabilities().supports_fp8 is True
