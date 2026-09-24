"""Tests for FP8 attention paths (Triton flash + scaled_mm)."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path as _Path

from repercep.attention import AttentionKind, AttentionShape
from repercep.attention.fp8_scaled_mm import FP8ScaledMMAttention
from repercep.attention.fp8_triton import FP8TritonAttention
from repercep.attention.naive import NaiveAttention
from repercep.attention.protocol import AttentionOp
from repercep.hardware import DType

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TRITON = importlib.util.find_spec("triton") is not None


def _gpu_or_skip() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")
    # The AMD FP8 ops below use the fp8_e4m3fnuz / fp8e4b8 dtypes — gfx942-only.
    # On NVIDIA hosts the kernel compile errors out ("type fp8e4b8 not supported
    # in this architecture"); the corresponding Hopper kernel lives in
    # tests/test_attention_cuda.py.  See ADR-0006 + F25 in BUILD_LOG.md.
    if not torch.version.hip:
        pytest.skip("fp8e4m3fnuz / fp8e4b8 paths are AMD-only; see tests/test_attention_cuda.py")


def test_fp8_scaled_mm_satisfies_protocol() -> None:
    op = FP8ScaledMMAttention()
    assert isinstance(op, AttentionOp)
    assert op.name == "fp8-scaled-mm"


def test_fp8_triton_satisfies_protocol() -> None:
    op = FP8TritonAttention()
    assert isinstance(op, AttentionOp)
    assert op.name == "fp8-triton-flash"


def test_fp8_scaled_mm_rejects_unsupported_kinds() -> None:
    op = FP8ScaledMMAttention()
    if not op.available:
        pytest.skip("torch._scaled_mm + fp8_e4m3fnuz unavailable")
    # NEIGHBORHOOD requires a mask in the scores matrix — not supported today.
    shape = AttentionShape(
        batch=1,
        heads=8,
        seq_len_q=512,
        seq_len_kv=512,
        head_dim=128,
        kind=AttentionKind.NEIGHBORHOOD,
    )
    assert not op.supports(shape, DType.BF16)


def test_fp8_triton_requires_min_seq_len() -> None:
    op = FP8TritonAttention()
    if not op.available:
        pytest.skip("triton FP8 kernel unavailable")
    # Below the min length the supports() must decline.
    small = AttentionShape(batch=1, heads=2, seq_len_q=64, seq_len_kv=64, head_dim=64)
    assert not op.supports(small, DType.BF16)
    ok = AttentionShape(batch=1, heads=2, seq_len_q=256, seq_len_kv=256, head_dim=64)
    assert op.supports(ok, DType.BF16)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_fp8_scaled_mm_matches_sdpa_within_fp8_tolerance() -> None:
    import torch

    _gpu_or_skip()
    op = FP8ScaledMMAttention()
    if not op.available:
        pytest.skip("torch._scaled_mm + fp8_e4m3fnuz unavailable")

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 4, 512, 128  # full set above _MIN_DIM and a multiple of 16
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)

    ref = NaiveAttention()(q, k, v)
    out = op(q, k, v)
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    # FP8 tolerance: max abs diff is dominated by the e4m3 quantization
    # step; we allow up to ~10x typical magnitude as the per-element ceiling.
    ref_scale = ref.float().abs().mean().item() + 1e-6
    diff = (out.float() - ref.float()).abs()
    assert diff.mean().item() / ref_scale < 0.30, (
        f"FP8 scaled_mm mean rel error {diff.mean().item() / ref_scale:.3f} too large"
    )


@pytest.mark.skipif(not _HAS_TORCH or not _HAS_TRITON, reason="torch+triton required")
def test_fp8_triton_matches_sdpa_within_fp8_tolerance() -> None:
    import torch

    _gpu_or_skip()
    op = FP8TritonAttention()
    if not op.available:
        pytest.skip("triton FP8 flash kernel unavailable")

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 4, 256, 64
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)

    ref = NaiveAttention()(q, k, v)
    out = op(q, k, v)
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    ref_scale = ref.float().abs().mean().item() + 1e-6
    diff = (out.float() - ref.float()).abs()
    # FP8 fused attention: per-tile quantization keeps the rel error in the
    # 1-5% range across typical magnitudes; we use a generous 15% cap for CI
    # stability.
    assert diff.mean().item() / ref_scale < 0.15


@pytest.mark.skipif(not _HAS_TORCH or not _HAS_TRITON, reason="torch+triton required")
def test_fp8_triton_causal_matches_sdpa() -> None:
    """Causal mask must match SDPA causal output within FP8 noise."""
    import torch

    _gpu_or_skip()
    op = FP8TritonAttention()
    if not op.available:
        pytest.skip("triton FP8 flash kernel unavailable")

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 4, 256, 64
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)

    ref = NaiveAttention()(q, k, v, causal=True)
    out = op(q, k, v, causal=True)
    ref_scale = ref.float().abs().mean().item() + 1e-6
    diff = (out.float() - ref.float()).abs()
    assert diff.mean().item() / ref_scale < 0.15


# ----- Autotune cache (CPU-only tests for the persistence layer) ------------


@pytest.mark.skipif(not _HAS_TRITON, reason="triton required to import the kernel module")
def test_fp8_autotune_cache_roundtrip(
    tmp_path: _Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache load/save round-trips through the kernel's helpers."""
    cache_path = tmp_path / "fp8_autotune.json"
    monkeypatch.setenv("REPERCEP_FP8_AUTOTUNE_CACHE", str(cache_path))

    import sys
    from pathlib import Path

    # Add kernels/ to sys.path so the kernel module is importable.
    kernels_dir = Path(__file__).resolve().parent.parent / "kernels"
    if str(kernels_dir) not in sys.path:
        sys.path.insert(0, str(kernels_dir))

    from triton_kernels.fp8_flash_attn import (
        _cache_key,
        _cache_load,
        _record_config,
        _resolve_config,
    )

    # Empty cache → empty dict and no resolved config.
    assert _cache_load() == {}
    assert _resolve_config(2, 32, 109120, 109120, 128, False) == {}

    # Round-trip a config.
    cfg = {"BLOCK_M": 256, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2}
    _record_config(2, 32, 109120, 109120, 128, False, cfg)

    # On-disk file exists, and the key uses the canonical encoding.
    assert cache_path.exists()
    data = _cache_load()
    assert _cache_key(2, 32, 109120, 109120, 128, False) in data
    assert _resolve_config(2, 32, 109120, 109120, 128, False) == cfg

    # A different shape misses the cache.
    assert _resolve_config(1, 8, 8192, 8192, 128, False) == {}

    # A separate save preserves earlier entries.
    other = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
    _record_config(1, 8, 8192, 8192, 128, False, other)
    full = _cache_load()
    assert len(full) == 2
    assert _resolve_config(2, 32, 109120, 109120, 128, False) == cfg
    assert _resolve_config(1, 8, 8192, 8192, 128, False) == other


@pytest.mark.skipif(not _HAS_TRITON, reason="triton required to import the kernel module")
def test_fp8_autotune_cache_corruption_is_ignored(
    tmp_path: _Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt JSON cache must not crash the kernel — treat as a cache miss."""
    cache_path = tmp_path / "fp8_autotune.json"
    cache_path.write_text("{ this is not valid json")
    monkeypatch.setenv("REPERCEP_FP8_AUTOTUNE_CACHE", str(cache_path))

    import sys
    from pathlib import Path

    kernels_dir = Path(__file__).resolve().parent.parent / "kernels"
    if str(kernels_dir) not in sys.path:
        sys.path.insert(0, str(kernels_dir))

    from triton_kernels.fp8_flash_attn import _cache_load, _resolve_config

    assert _cache_load() == {}
    assert _resolve_config(2, 32, 109120, 109120, 128, False) == {}


@pytest.mark.skipif(not _HAS_TRITON, reason="triton required to import the kernel module")
def test_fp8_autotune_grid_constraints() -> None:
    """The autotune search grid satisfies the documented constraints."""
    import sys
    from pathlib import Path

    kernels_dir = Path(__file__).resolve().parent.parent / "kernels"
    if str(kernels_dir) not in sys.path:
        sys.path.insert(0, str(kernels_dir))

    from triton_kernels.fp8_flash_attn import _AUTOTUNE_CONFIGS

    assert len(_AUTOTUNE_CONFIGS) > 0
    for cfg in _AUTOTUNE_CONFIGS:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        # MFMA tile floor.
        assert bm >= 32 and bn >= 32
        # Documented BLOCK_N <= BLOCK_M*2 (avoid pathological LDS layouts).
        assert bn <= bm * 2
        # Each tile bounded by LDS budget.
        assert bm * bn <= 256 * 256
        assert cfg.num_warps in (4, 8, 16)
        assert cfg.num_stages in (2, 3)
