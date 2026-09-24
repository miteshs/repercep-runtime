"""Tests for the NVIDIA attention ops.  Most are skipped without CUDA.

Sibling of ``test_attention.py`` (which exercises the AMD ops); follows the
same pattern of unconditional structural checks plus GPU-gated functional
checks.
"""

from __future__ import annotations

import importlib.util

import pytest

from repercep.attention import AttentionShape, select_attention_op
from repercep.attention.fp8_hopper_triton import FP8HopperTritonAttention
from repercep.attention.hopper_flash import HopperFlashAttention
from repercep.attention.naive import NaiveAttention
from repercep.attention.protocol import AttentionOp
from repercep.attention.transformer_engine import TransformerEngineAttention
from repercep.backend.cuda import CUDABackend
from repercep.hardware import H100, DType

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_hopper_flash_satisfies_protocol() -> None:
    assert isinstance(HopperFlashAttention(), AttentionOp)


def test_fp8_hopper_triton_satisfies_protocol() -> None:
    assert isinstance(FP8HopperTritonAttention(), AttentionOp)


def test_transformer_engine_satisfies_protocol() -> None:
    assert isinstance(TransformerEngineAttention(), AttentionOp)


def test_hopper_flash_op_name() -> None:
    # Stable name for diagnostic surfaces (repercep info, registry traces).
    assert HopperFlashAttention().name == "nvidia-flash"


def test_fp8_hopper_triton_op_name() -> None:
    assert FP8HopperTritonAttention().name == "fp8-hopper-triton-flash"


def test_transformer_engine_op_name() -> None:
    assert TransformerEngineAttention().name == "transformer-engine-fp8"


def test_select_hopper_returns_attention_op() -> None:
    # Without flash-attn, FP8 env, or TE installed, selection falls to naive.
    shape = AttentionShape(batch=1, heads=16, seq_len_q=1024, seq_len_kv=1024, head_dim=128)
    op = select_attention_op(H100, shape, DType.BF16)
    assert isinstance(op, AttentionOp)
    # Names that can appear when the FP8 env var is unset and flash-attn is
    # not built: TE is also unconditional in the FP8 branch when env is on,
    # so its name should not appear here.
    assert op.name in ("nvidia-flash", "naive-sdpa")


def test_fp8_hopper_triton_supports_requires_min_seqlen() -> None:
    # Even when the kernel itself is unavailable, supports() should be False
    # without it.  When importable, the min-seqlen gate applies.
    op = FP8HopperTritonAttention()
    short_shape = AttentionShape(batch=1, heads=8, seq_len_q=64, seq_len_kv=64, head_dim=64)
    assert op.supports(short_shape, DType.BF16) is False


def test_fp8_hopper_triton_supports_cross_attention_false() -> None:
    op = FP8HopperTritonAttention()
    cross = AttentionShape(batch=1, heads=8, seq_len_q=4096, seq_len_kv=2048, head_dim=128)
    # Self-attention only for now.
    assert op.supports(cross, DType.BF16) is False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_naive_runs_on_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")
    if not (torch.version.cuda is not None and not torch.version.hip):
        pytest.skip("not a CUDA host")
    dev = torch.device("cuda", 0)
    b, h, s, d = 2, 8, 256, 64
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    out = NaiveAttention()(q, k, v, causal=True)
    assert out.shape == (b, h, s, d)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_hopper_flash_runs_when_built() -> None:
    # Skipped unless flash-attn is installed; structural gate on availability.
    op = HopperFlashAttention()
    if not op.available:
        pytest.skip("flash-attn not installed (see docs/COSMOS_ON_H100.md)")
    import torch

    dev = CUDABackend().torch_device(0)
    b, h, s, d = 1, 16, 512, 128
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = op(q, k, v)
    assert tuple(out.shape) == (b, h, s, d)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_hopper_flash_matches_sdpa() -> None:
    """FA-2 / FA-3 output must match SDPA within BF16 numerics on H100.

    SDPA on Hopper already routes to cuDNN flash-attn, so this is a true
    parity check, not a "flash is better" check.  The wrapper exists to
    expose the FA-3 entry point when it's built; both paths should agree
    on output bytes.
    """
    op = HopperFlashAttention()
    if not op.available:
        pytest.skip("flash-attn not installed")
    import torch

    dev = CUDABackend().torch_device(0)
    b, h, s, d = 1, 16, 4096, 128
    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn_like(q) / 8.0
    v = torch.randn_like(q)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = op(q, k, v)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel.item() < 0.01, (
        f"HopperFlash vs SDPA rel diff {rel.item():.4f} > 0.01 — "
        f"is_fa3={op.is_fa3}, suggests a layout or numerics regression"
    )


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_fp8_hopper_triton_matches_sdpa() -> None:
    """FP8 Hopper Triton kernel must match SDPA within FP8 tolerance on H100.

    Same correctness contract as the AMD ``test_fp8_triton_matches_sdpa``
    in ``tests/test_attention_fp8.py``, run on the sibling Hopper kernel.
    Tolerance is wider because FP8 quantization is unavoidably lossy
    (~3-5% mean rel diff at the Cosmos production shape on Hopper —
    see F27 in docs/BUILD_LOG.md).
    """
    op = FP8HopperTritonAttention()
    if not op.available:
        pytest.skip("Triton FP8 Hopper kernel unavailable")
    import torch

    dev = CUDABackend().torch_device(0)
    b, h, s, d = 1, 8, 4096, 128
    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn_like(q) / 8.0
    v = torch.randn_like(q)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = op(q, k, v)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    # F27 measured 3.4% at this shape; allow headroom for autotune drift.
    assert rel.item() < 0.10, (
        f"FP8 Hopper Triton vs SDPA rel diff {rel.item():.4f} > 0.10 — "
        f"either the kernel regressed or autotune picked a degenerate config"
    )
