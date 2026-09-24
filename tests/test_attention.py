"""Tests for attention op selection and correctness."""

from __future__ import annotations

import importlib.util

import pytest

from repercep.attention import AttentionKind, AttentionShape, select_attention_op
from repercep.attention.naive import NaiveAttention
from repercep.attention.protocol import AttentionOp
from repercep.backend.rocm import ROCmBackend
from repercep.hardware import MI300X, DType

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_naive_supports_any_shape() -> None:
    # The naive op is the correctness floor: it must never decline a shape.
    odd = AttentionShape(
        batch=1,
        heads=1,
        seq_len_q=7,
        seq_len_kv=13,
        head_dim=999,
        kind=AttentionKind.NEIGHBORHOOD,
    )
    assert NaiveAttention().supports(odd, DType.FP32)


def test_naive_declares_available() -> None:
    op = NaiveAttention()
    assert isinstance(op, AttentionOp)
    assert op.available is True


def test_select_returns_attention_op() -> None:
    shape = AttentionShape(batch=1, heads=16, seq_len_q=1024, seq_len_kv=1024, head_dim=128)
    op = select_attention_op(MI300X, shape, DType.BF16)
    assert isinstance(op, AttentionOp)
    assert isinstance(op.available, bool)
    # Without the CK flash-attn build installed, selection falls to the floor.
    assert op.name in ("rocm-ck-flash", "naive-sdpa")


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_naive_attention_runs() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")
    dev = torch.device("cuda", 0)
    b, h, s, d = 2, 8, 256, 64
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    out = NaiveAttention()(q, k, v, causal=True)
    assert out.shape == (b, h, s, d)
    assert torch.isfinite(out).all()


def test_backend_attention_op_runs_cosmos_dit_shape() -> None:
    torch = pytest.importorskip("torch")
    backend = ROCmBackend()
    if not backend.is_available():
        pytest.skip("no ROCm GPU on host")
    # Cosmos-Predict-7B DiT self-attention: 32 heads x head_dim 128. On ROCm,
    # the selected op routes through SDPA -> aotriton flash kernels.
    shape = AttentionShape(batch=1, heads=32, seq_len_q=512, seq_len_kv=512, head_dim=128)
    op = backend.attention_op(shape, DType.BF16)
    device = backend.torch_device(0)
    q = torch.randn(1, 32, 512, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = op(q, k, v)
    assert tuple(out.shape) == (1, 32, 512, 128)
    assert torch.isfinite(out).all()
