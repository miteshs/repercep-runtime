"""Tests for the Ada Lovelace (sm_89) FP8 Triton flash-attention op.

Sibling of ``test_attention_cuda.py::test_fp8_hopper_triton_matches_sdpa``
(Hopper / sm_90a) and ``test_attention_fp8.py::test_fp8_triton_matches_sdpa``
(CDNA3 / gfx942).  All three kernels share the FlashAttention-2 online-
softmax algorithm; this test confirms the Ada sibling produces output
within FP8 quantization noise of the reference SDPA on the spec'd
``(1, 8, 4096, 128)`` shape from the BUILD_LOG (Item J).

Most tests are skipped unless the host has an Ada Lovelace GPU
(capability (8, 9)).  The structural tests (protocol, name, supports())
run unconditionally — on a non-Ada host the wrapper disqualifies itself
at __init__ time, so ``available`` is False and ``supports()`` returns
False, and we assert those negatives.
"""

from __future__ import annotations

import importlib.util

import pytest

from repercep.attention.fp8_ada_triton import FP8AdaTritonAttention
from repercep.attention.protocol import AttentionOp
from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _is_ada() -> bool:
    """True iff we're on an Ada Lovelace GPU."""
    if not _HAS_TORCH:
        return False
    import torch

    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (8, 9)


# --- Structural tests (run anywhere) ----------------------------------------


def test_fp8_ada_triton_satisfies_protocol() -> None:
    assert isinstance(FP8AdaTritonAttention(), AttentionOp)


def test_fp8_ada_triton_op_name() -> None:
    # Stable name for diagnostic surfaces (repercep info, registry traces).
    assert FP8AdaTritonAttention().name == "fp8-ada-triton-flash"


def test_fp8_ada_triton_disqualifies_off_ada() -> None:
    """On a non-Ada host (or no GPU), the wrapper must self-disqualify.

    This is the silicon gate — the kernel uses sm_89 FP8 mma.sync that
    will not compile on Ampere (sm_80/86), and on Hopper the dedicated
    Hopper kernel is faster.  We assert the wrapper says so cleanly
    instead of letting Triton emit a confusing arch error at first call.
    """
    op = FP8AdaTritonAttention()
    if _is_ada():
        # If we're on Ada, op.available depends on whether triton imports
        # the kernel cleanly — that's a separate concern, exercised below.
        pytest.skip("on Ada host; this test exercises the off-Ada path")
    assert op.available is False, (
        "FP8AdaTritonAttention.available must be False on non-Ada hosts; "
        "registry would otherwise try to route to a kernel that won't compile."
    )
    # supports() should also be False regardless of shape/dtype.
    shape = AttentionShape(batch=1, heads=8, seq_len_q=4096, seq_len_kv=4096, head_dim=128)
    assert op.supports(shape, DType.BF16) is False


def test_fp8_ada_triton_supports_short_seq_false() -> None:
    # Even on Ada with a working kernel, short seq → False (perf cutoff).
    op = FP8AdaTritonAttention()
    short = AttentionShape(batch=1, heads=8, seq_len_q=64, seq_len_kv=64, head_dim=64)
    assert op.supports(short, DType.BF16) is False


def test_fp8_ada_triton_supports_cross_attention_false() -> None:
    op = FP8AdaTritonAttention()
    cross = AttentionShape(batch=1, heads=8, seq_len_q=4096, seq_len_kv=2048, head_dim=128)
    assert op.supports(cross, DType.BF16) is False


# --- Functional tests (Ada hardware required) -------------------------------


@pytest.mark.skipif(not _is_ada(), reason="requires Ada Lovelace (sm_89) GPU")
def test_fp8_ada_triton_available_on_ada() -> None:
    """On Ada we expect the kernel to import successfully."""
    op = FP8AdaTritonAttention()
    assert op.available, (
        f"FP8AdaTritonAttention should be available on Ada; "
        f"import_error={op._import_error!r}"
    )


@pytest.mark.skipif(not _is_ada(), reason="requires Ada Lovelace (sm_89) GPU")
def test_fp8_ada_triton_supports_spec_shape() -> None:
    """Spec'd shape (1, 8, 4096, 128) BF16 must be supported on Ada."""
    op = FP8AdaTritonAttention()
    if not op.available:
        pytest.skip(f"kernel did not import: {op._import_error}")
    shape = AttentionShape(
        batch=1, heads=8, seq_len_q=4096, seq_len_kv=4096, head_dim=128,
        kind=AttentionKind.FULL,
    )
    assert op.supports(shape, DType.BF16) is True


@pytest.mark.skipif(not _is_ada(), reason="requires Ada Lovelace (sm_89) GPU")
def test_fp8_ada_triton_matches_sdpa() -> None:
    """FP8 Ada Triton kernel must match SDPA within FP8 tolerance on sm_89.

    Same correctness contract as the Hopper sibling
    (``test_fp8_hopper_triton_matches_sdpa``); tolerance is the standard
    5-10% FP8 quantization noise floor.  We measure on the spec'd
    (1, 8, 4096, 128) shape used to greenlight the Ada port.
    """
    op = FP8AdaTritonAttention()
    if not op.available:
        pytest.skip(f"Triton FP8 Ada kernel unavailable: {op._import_error}")

    import torch

    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 8, 4096, 128
    torch.manual_seed(0)
    # The /8 scaling mirrors the Hopper test — keeps Q@K^T inside FP32 range
    # so the SDPA reference doesn't itself overflow before FP8 quantization
    # noise dominates.
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn_like(q) / 8.0
    v = torch.randn_like(q)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = op(q, k, v)

    assert tuple(out.shape) == (b, h, s, d)
    assert torch.isfinite(out).all(), "FP8 Ada attention produced non-finite values"

    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    # 10% headroom — FP8 quantization noise on Ada is ~5% at this shape
    # (see docs/FP8_ON_ADA.md); the wider bound covers autotune drift.
    assert rel.item() < 0.10, (
        f"FP8 Ada Triton vs SDPA rel diff {rel.item():.4f} > 0.10 — "
        f"either the kernel regressed or autotune picked a degenerate config"
    )


@pytest.mark.skipif(not _is_ada(), reason="requires Ada Lovelace (sm_89) GPU")
def test_fp8_ada_triton_head_dim_64() -> None:
    """Smaller head_dim (64) must also work — many DiT variants use 64."""
    op = FP8AdaTritonAttention()
    if not op.available:
        pytest.skip(f"Triton FP8 Ada kernel unavailable: {op._import_error}")

    import torch

    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 8, 4096, 64
    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn_like(q) / 8.0
    v = torch.randn_like(q)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = op(q, k, v)

    assert tuple(out.shape) == (b, h, s, d)
    assert torch.isfinite(out).all()

    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel.item() < 0.10, f"D=64 rel diff {rel.item():.4f} > 0.10"


@pytest.mark.skipif(not _is_ada(), reason="requires Ada Lovelace (sm_89) GPU")
def test_fp8_ada_triton_causal() -> None:
    """Causal masking must produce a strictly lower-triangular attention."""
    op = FP8AdaTritonAttention()
    if not op.available:
        pytest.skip(f"Triton FP8 Ada kernel unavailable: {op._import_error}")

    import torch

    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 8, 4096, 128
    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn_like(q) / 8.0
    v = torch.randn_like(q)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = op(q, k, v, causal=True)

    assert tuple(out.shape) == (b, h, s, d)
    assert torch.isfinite(out).all()

    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel.item() < 0.10, f"causal rel diff {rel.item():.4f} > 0.10"
