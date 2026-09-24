"""Tests for the NVIDIA CUDA backend.  GPU-dependent tests skip without CUDA.

Sibling of ``test_backend.py`` (which exercises ROCmBackend).  The two are
intentionally parallel — same shape of tests, same protocol conformance, same
skip pattern — because ADR-0006 commits to the symmetry between the two
backends.
"""

from __future__ import annotations

import pytest

from repercep.backend.cuda import CUDABackend
from repercep.backend.protocol import Backend
from repercep.backend.registry import select_backend
from repercep.hardware import DType, Vendor


def test_cuda_backend_identity() -> None:
    backend = CUDABackend()
    assert backend.vendor is Vendor.NVIDIA
    assert backend.name == "cuda"


def test_cuda_backend_satisfies_protocol() -> None:
    # Structural conformance — the whole point of ADR-0003's Protocol seam.
    assert isinstance(CUDABackend(), Backend)


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_cuda_devices_detected() -> None:
    backend = CUDABackend()
    devices = backend.devices()
    assert len(devices) >= 1
    # CUDA arch ids start with "sm" on NVIDIA (vs "gfx" on AMD).
    assert devices[0].arch.gfx_id.startswith("sm")
    assert devices[0].arch.vendor is Vendor.NVIDIA
    assert devices[0].total_memory_bytes > 0
    assert devices[0].multi_processor_count > 0


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_select_backend_returns_cuda_when_no_rocm() -> None:
    # On a pure-CUDA host (no ROCm wheel), select_backend should return CUDA.
    # On a dual-vendor host (rare), ROCm wins per the registry ordering.
    chosen = select_backend()
    assert chosen.name in ("cuda", "rocm")


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_select_cuda_explicitly() -> None:
    assert select_backend(prefer="cuda").name == "cuda"


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_capabilities_hopper_advertises_fp8() -> None:
    # Hopper (sm_90 / sm_90a) and Ada (sm_89) have hardware FP8.  This host's
    # device is Hopper per the H100 confirmation — capabilities should
    # advertise both FP8 dtypes.
    caps = CUDABackend().capabilities()
    backend = CUDABackend()
    arch = backend.devices()[0].arch.gfx_id
    if arch.startswith(("sm90", "sm89")):
        assert caps.supports_fp8 is True
        assert DType.FP8_E4M3 in caps.dtypes
        assert DType.FP8_E5M2 in caps.dtypes


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_torch_device_handle() -> None:
    import torch

    dev = CUDABackend().torch_device(0)
    assert dev.type == "cuda"
    assert dev.index == 0
    # Round-trip: allocate on it and confirm.
    t = torch.zeros(4, device=dev)
    assert t.device.type == "cuda"


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_default_dtype_is_bf16() -> None:
    # BF16 is the reference dtype for Cosmos and Wan; matches ROCmBackend.
    assert CUDABackend().default_dtype() is DType.BF16


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_attention_op_runs_cosmos_dit_shape() -> None:
    import torch

    from repercep.attention.types import AttentionShape

    backend = CUDABackend()
    # Cosmos-Predict-7B DiT self-attention: 32 heads x head_dim 128.  On
    # CUDA, the selected op routes through SDPA (the floor) when neither
    # flash-attn nor the FP8 env var is set.
    shape = AttentionShape(batch=1, heads=32, seq_len_q=512, seq_len_kv=512, head_dim=128)
    op = backend.attention_op(shape, DType.BF16)
    device = backend.torch_device(0)
    q = torch.randn(1, 32, 512, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = op(q, k, v)
    assert tuple(out.shape) == (1, 32, 512, 128)
    assert torch.isfinite(out).all()
