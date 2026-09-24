"""Tests for ``scripts/compute_fvd.py`` — Fréchet Video Distance.

Most assertions target the math layer (``_frechet_distance``,
``_per_clip_l2_fallback``, ``_video_to_clips``) so they run instantly without
a model download.  A single end-to-end test exercises the full ``compute_fvd``
pipeline (which loads I3D from the torch.hub cache) — skipped if the I3D
weight file is not already cached, so unit tests stay offline-friendly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# scripts/compute_fvd.py isn't an importable package; load by path.
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
_FVD_SRC = _SCRIPTS / "compute_fvd.py"

_spec = importlib.util.spec_from_file_location("compute_fvd_mod", _FVD_SRC)
assert _spec is not None and _spec.loader is not None
_fvd = importlib.util.module_from_spec(_spec)
sys.modules["compute_fvd_mod"] = _fvd
_spec.loader.exec_module(_fvd)


# Path the i3d_r50 pretrained checkpoint lands at via torch.hub.
_I3D_CACHE = (
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "I3D_8x8_R50.pyth"
)


def test_frechet_identity_is_near_zero() -> None:
    """FVD(A, A) should be ~0 modulo sqrtm numerical noise."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(0)
    # 8 samples in 32-D so covariance is well-defined.
    feats = rng.standard_normal((8, 32)).astype(np.float64)
    d = _fvd._frechet_distance(feats, feats)
    # sqrtm of (Σ @ Σ) can carry a few ulps of error; threshold generously.
    assert d < 1e-4, f"FVD(A,A) = {d}, expected ~0"


def test_frechet_positive_for_different_distributions() -> None:
    """Two different distributions should give a strictly positive FVD."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(0)
    feats_a = rng.standard_normal((16, 32)).astype(np.float64)
    feats_b = (rng.standard_normal((16, 32)) + 3.0).astype(np.float64)  # shifted
    d = _fvd._frechet_distance(feats_a, feats_b)
    assert d > 0, f"FVD between different distributions should be > 0, got {d}"


def test_per_clip_l2_fallback_identity() -> None:
    """L2 fallback over identical feature vectors should be 0."""
    np = pytest.importorskip("numpy")
    feats = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    assert _fvd._per_clip_l2_fallback(feats, feats) == 0.0


def test_per_clip_l2_fallback_known_value() -> None:
    """L2 fallback for ``[0,0,0]`` vs ``[3,4,0]`` should equal 5.0."""
    np = pytest.importorskip("numpy")
    a = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    b = np.array([[3.0, 4.0, 0.0]], dtype=np.float64)
    assert _fvd._per_clip_l2_fallback(a, b) == pytest.approx(5.0)


def test_video_to_clips_shape() -> None:
    """``_video_to_clips`` should yield ``(num_clips, 3, 8, 224, 224)``."""
    np = pytest.importorskip("numpy")
    # 30-frame 480x640 uint8 synthetic video.
    frames = (np.random.rand(30, 480, 640, 3) * 255).astype(np.uint8)
    clips = _fvd._video_to_clips(frames, num_clips=1)
    assert clips.shape == (1, 3, 8, 224, 224)
    assert clips.dtype == np.float32
    clips_multi = _fvd._video_to_clips(frames, num_clips=3)
    assert clips_multi.shape == (3, 3, 8, 224, 224)


def test_video_to_clips_rejects_too_short() -> None:
    """Videos shorter than 8 frames should raise — I3D needs T >= 8."""
    np = pytest.importorskip("numpy")
    frames = (np.random.rand(5, 240, 320, 3) * 255).astype(np.uint8)
    with pytest.raises(ValueError, match="need >= 8"):
        _fvd._video_to_clips(frames, num_clips=1)


def test_compute_fvd_rejects_empty_inputs(tmp_path: Path) -> None:
    """No reference / no candidates should fail with a clear ValueError."""
    sample = tmp_path / "fake.mp4"
    sample.write_bytes(b"")
    with pytest.raises(ValueError, match="reference"):
        _fvd.compute_fvd([], [sample], device="cpu")
    with pytest.raises(ValueError, match="candidates"):
        _fvd.compute_fvd([sample], [], device="cpu")


def test_compute_fvd_n_warn_fires(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """N-warn should fire loudly when min(N) < threshold (but >= 2)."""
    if not _I3D_CACHE.exists():
        pytest.skip("I3D weights not cached; skipping end-to-end FVD test")
    np = pytest.importorskip("numpy")
    pytest.importorskip("imageio.v3")
    pytest.importorskip("scipy.linalg")
    pytest.importorskip("torch")

    import imageio.v3 as iio

    # Two small mp4s on each side -> N_clips = 2 each (still < 50).
    refs: list[Path] = []
    cands: list[Path] = []
    for i in range(2):
        p = tmp_path / f"ref_{i}.mp4"
        frames = (np.random.default_rng(i).random((20, 96, 96, 3)) * 255).astype(
            np.uint8
        )
        iio.imwrite(p, frames, plugin="FFMPEG", fps=24, codec="libx264")
        refs.append(p)
    for i in range(2):
        p = tmp_path / f"cand_{i}.mp4"
        frames = (
            np.random.default_rng(100 + i).random((20, 96, 96, 3)) * 255
        ).astype(np.uint8)
        iio.imwrite(p, frames, plugin="FFMPEG", fps=24, codec="libx264")
        cands.append(p)

    capsys.readouterr()  # drain
    out = _fvd.compute_fvd(refs, cands, device="cpu", num_clips=1, n_warn=50)
    captured = capsys.readouterr()

    assert out["mode"] == "fvd"
    assert isinstance(out["fvd"], float)
    assert out["n_reference_clips"] == 2
    assert out["n_candidate_clips"] == 2
    assert out["feature_dim"] == 2048
    # LOUD WARNING should land on stderr.
    assert "LOUD WARNING" in captured.err
    assert "N >= 50" in captured.err or "N >= 1000" in captured.err


def test_compute_fvd_n1_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """N=1 vs N=1 should fall back to per-clip L2 (FVD undefined)."""
    if not _I3D_CACHE.exists():
        pytest.skip("I3D weights not cached; skipping end-to-end FVD test")
    np = pytest.importorskip("numpy")
    pytest.importorskip("imageio.v3")
    pytest.importorskip("torch")

    import imageio.v3 as iio

    ref = tmp_path / "ref.mp4"
    cand = tmp_path / "cand.mp4"
    iio.imwrite(
        ref,
        (np.random.default_rng(0).random((20, 96, 96, 3)) * 255).astype(np.uint8),
        plugin="FFMPEG",
        fps=24,
        codec="libx264",
    )
    iio.imwrite(
        cand,
        (np.random.default_rng(1).random((20, 96, 96, 3)) * 255).astype(np.uint8),
        plugin="FFMPEG",
        fps=24,
        codec="libx264",
    )

    capsys.readouterr()  # drain
    out = _fvd.compute_fvd([ref], [cand], device="cpu", num_clips=1)
    captured = capsys.readouterr()

    assert out["mode"] == "feature_l2"
    assert isinstance(out["feature_l2"], float)
    assert out["n_reference_clips"] == 1
    assert out["n_candidate_clips"] == 1
    assert "too small for FVD" in captured.err
