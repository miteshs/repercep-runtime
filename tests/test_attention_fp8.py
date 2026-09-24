"""Tests for FP8 attention paths (Triton flash + scaled_mm)."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

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


# ----- AMD backend knobs (matrix_instr_nonkdim / kpack / waves_per_eu) ------
#
# These reach Triton's HIP backend as HIPOptions fields, passed inside the
# Config kwargs dict.  The default grid must stay byte-identical so the
# initial-tune tax quoted in docs/OPTIMIZATION.md — and every Cosmos number
# measured under it — remains comparable; the sweep is opt-in.


def _kernel_module() -> Any:
    import sys
    from pathlib import Path

    kernels_dir = Path(__file__).resolve().parent.parent / "kernels"
    if str(kernels_dir) not in sys.path:
        sys.path.insert(0, str(kernels_dir))
    from triton_kernels import fp8_flash_attn

    return fp8_flash_attn


@pytest.mark.skipif(not _HAS_TRITON, reason="triton required to import the kernel module")
def test_amd_knob_sweep_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _kernel_module()
    monkeypatch.delenv("REPERCEP_FP8_TUNE_AMD_KNOBS", raising=False)
    assert not mod._tune_amd_knobs_enabled()
    # The module-level grid was built with the sweep off, so no config in it
    # may carry an AMD knob.
    for cfg in mod._AUTOTUNE_CONFIGS:
        assert not (set(cfg.kwargs) & set(mod._AMD_KNOB_NAMES))


@pytest.mark.skipif(not _HAS_TRITON, reason="triton required to import the kernel module")
def test_amd_knob_support_is_introspected_not_assumed() -> None:
    """The knob set must come from the installed backend, not a hardcoded list.

    ``kpack`` is deprecated on gfx950 and the field set has churned across
    ROCm releases; passing an undeclared option raises at launch.  On a
    CUDA-only or CPU-only install this is legitimately empty.
    """
    mod = _kernel_module()
    supported = mod._supported_amd_knobs()
    assert isinstance(supported, tuple)
    assert set(supported) <= set(mod._AMD_KNOB_NAMES)


@pytest.mark.skipif(not _HAS_TRITON, reason="triton required to import the kernel module")
def test_amd_knob_sweep_extends_grid_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _kernel_module()
    baseline = len(mod._autotune_configs())

    monkeypatch.setenv("REPERCEP_FP8_TUNE_AMD_KNOBS", "1")
    extended = mod._autotune_configs()

    if not mod._supported_amd_knobs():
        # No HIP backend on this host: the sweep is a no-op rather than an error.
        assert len(extended) == baseline
        pytest.skip("no AMD backend knobs declared by this triton install")

    assert len(extended) > baseline
    # Bounded: anchored to two tile shapes, so the tune tax stays ~2x, not ~12x.
    assert len(extended) <= baseline * 3
    knobbed = [c for c in extended if set(c.kwargs) & set(mod._AMD_KNOB_NAMES)]
    assert knobbed, "sweep enabled but no config carries a knob"
    for cfg in knobbed:
        assert (cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"]) in ((128, 64), (256, 128))
        # One knob at a time — a full cross product is what blows up the tax.
        assert len(set(cfg.kwargs) & set(mod._AMD_KNOB_NAMES)) == 1


@pytest.mark.skipif(not _HAS_TRITON, reason="triton required to import the kernel module")
def test_amd_knobs_round_trip_through_the_cache(
    tmp_path: _Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tuned knob must survive to the fixed-config launch path.

    Without this the second process silently reverts to backend defaults and
    the cached "winner" is not the configuration that actually won.
    """
    cache_path = tmp_path / "fp8_autotune.json"
    monkeypatch.setenv("REPERCEP_FP8_AUTOTUNE_CACHE", str(cache_path))
    mod = _kernel_module()

    cfg = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "num_warps": 4,
        "num_stages": 2,
        "matrix_instr_nonkdim": 32,
    }
    mod._record_config(1, 8, 8192, 8192, 128, False, cfg)
    resolved = mod._resolve_config(1, 8, 8192, 8192, 128, False)
    assert resolved == cfg
    assert {k: resolved[k] for k in mod._AMD_KNOB_NAMES if k in resolved} == {
        "matrix_instr_nonkdim": 32
    }


@pytest.mark.skipif(not _HAS_TRITON, reason="triton required to import the kernel module")
def test_pre_knob_cache_entries_stay_valid(
    tmp_path: _Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache files written before the knobs existed must still resolve.

    Absent keys mean "backend default", so no schema bump is needed and no
    user loses their tuned shapes on upgrade.
    """
    cache_path = tmp_path / "fp8_autotune.json"
    monkeypatch.setenv("REPERCEP_FP8_AUTOTUNE_CACHE", str(cache_path))
    mod = _kernel_module()

    legacy = {"BLOCK_M": 256, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2}
    mod._record_config(2, 32, 109120, 109120, 128, False, legacy)
    resolved = mod._resolve_config(2, 32, 109120, 109120, 128, False)
    assert resolved == legacy
    assert not (set(resolved) & set(mod._AMD_KNOB_NAMES))
