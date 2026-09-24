"""CPU/CUDA parity tests for the pixel-quality eval pipeline (Item I).

Item F (``scripts/eval_cpu_quality.py``) forces ``--device cpu`` on both
inner stages — LPIPS+MSE+PSNR and FVD.  That choice is methodological
(eval should be reproducible without GPU silicon), not a correctness
constraint: the underlying pixel-metric math should produce numerically
equivalent answers on either device.  This module asserts exactly that.

Scope:
  * MSE / PSNR — pure numpy, no device involved; assert bit-identical.
  * LPIPS (AlexNet backbone) — fp32 forward.  The correctness threshold
    is ``atol=1e-4`` (anything beyond that signals a real numerical
    issue per Item I's spec).  On this host we observe ~1.5e-5 max
    absolute drift, comfortably under the threshold and consistent with
    cuDNN picking different conv2d algorithms vs the CPU MKL path.

Explicitly out of scope:
  * FVD parity.  The I3D backbone (``compute_fvd.py``) pulls weights from
    Facebook's pytorchvideo URL; a cold cache on an offline VM stalls the
    test.  Marked ``pytest.mark.skip`` below.

Tests skip cleanly on CPU-only hosts via ``torch.cuda.is_available()``.

Companion to ``tests/test_eval_cpu_quality.py`` (mocked-stage CLI tests).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import types
    from collections.abc import Callable

    import numpy as np

# ---------------------------------------------------------------------------
# Optional-dep gates — keep import time cheap on CPU-only / minimal hosts.
# ---------------------------------------------------------------------------

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_LPIPS = importlib.util.find_spec("lpips") is not None
_HAS_IMAGEIO = importlib.util.find_spec("imageio") is not None

if _HAS_TORCH:
    import torch
    _HAS_CUDA = bool(torch.cuda.is_available())
else:  # pragma: no cover - exercised on torch-less hosts only
    _HAS_CUDA = False

_REQUIRES_CUDA = pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_CUDA and _HAS_LPIPS and _HAS_IMAGEIO),
    reason="parity test requires torch+CUDA+lpips+imageio",
)


# ---------------------------------------------------------------------------
# Load scripts/verify_quality.py by path — scripts/ isn't an importable package.
# Mirrors the loader pattern in tests/test_eval_cpu_quality.py + tests/test_fvd.py.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
_VQ_SRC = _SCRIPTS / "verify_quality.py"
_EVAL_SRC = _SCRIPTS / "eval_cpu_quality.py"


def _load_script(name: str, src: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic video fixtures — two 8-frame 64x64 mp4s with a small fixed delta.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_video_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two tiny mp4s: A = seeded uint8 noise, B = A + 5 (clipped).

    Small (8 frames, 64x64) so LPIPS forward is sub-second per frame even
    on CPU.  The +5 perturbation gives a non-trivial LPIPS distance — not
    zero (which would skip the model), not saturated near 1.  Module-scoped
    so we only encode once per test session.
    """
    if not _HAS_IMAGEIO:
        pytest.skip("imageio not installed")
    import imageio.v3 as iio
    import numpy as np

    tmp = tmp_path_factory.mktemp("eval_parity")
    rng = np.random.default_rng(seed=0xC0FFEE)
    a = rng.integers(0, 256, size=(8, 64, 64, 3), dtype=np.uint8)
    # +5 per pixel, clipped to keep uint8 valid.  Small enough to stay in
    # the perceptually-similar band — exactly what we want for an LPIPS
    # parity test (not too far, not zero).
    b = np.clip(a.astype(np.int16) + 5, 0, 255).astype(np.uint8)

    video_a = tmp / "a.mp4"
    video_b = tmp / "b.mp4"
    # macro_block_size=1 lets us encode arbitrary 64x64 frames without
    # ffmpeg complaining about the size not being a multiple of 16.
    iio.imwrite(video_a, a, plugin="FFMPEG", fps=8, macro_block_size=1)
    iio.imwrite(video_b, b, plugin="FFMPEG", fps=8, macro_block_size=1)
    return video_a, video_b


# ---------------------------------------------------------------------------
# Pixel-metric helpers — replicate the verify_quality formulas verbatim so
# we can drive them on an arbitrary device without going through the CLI.
# ---------------------------------------------------------------------------


def _mse_psnr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised MSE / PSNR per frame — identical formulas to verify_quality."""
    import numpy as np

    diff = a.astype(np.int32) - b.astype(np.int32)
    mse_per_frame = (diff.astype(np.float64) ** 2).mean(axis=(1, 2, 3))
    psnr_per_frame = np.where(
        mse_per_frame > 0,
        10.0
        * np.log10(
            (255.0 ** 2) / np.where(mse_per_frame > 0, mse_per_frame, 1.0)
        ),
        float("inf"),
    )
    return mse_per_frame, psnr_per_frame


def _lpips_per_frame_on_device(a: np.ndarray, b: np.ndarray, device: str) -> list[float]:
    """Run LPIPS(alex) per-frame, returning the scalar distance list."""
    import lpips
    import torch

    vq = _load_script("verify_quality_parity_mod", _VQ_SRC)
    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    loss_fn.eval()

    out: list[float] = []
    with torch.inference_mode():
        for i in range(a.shape[0]):
            ta = vq._to_lpips_tensor(a[i], device)
            tb = vq._to_lpips_tensor(b[i], device)
            d = loss_fn(ta, tb).item()
            out.append(float(d))
            # Eager release — caller wants no GPU memory held across frames.
            del ta, tb

    del loss_fn
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# 1) MSE / PSNR — bit-identical (pure numpy, no model, no device).
# ---------------------------------------------------------------------------


@_REQUIRES_CUDA
def test_mse_psnr_are_bit_identical(synthetic_video_pair: tuple[Path, Path]) -> None:
    """MSE and PSNR never touch the GPU; recomputing twice must be identical."""
    import numpy as np

    vq = _load_script("verify_quality_parity_mod", _VQ_SRC)
    video_a, video_b = synthetic_video_pair
    a = vq._read_video_frames(video_a)
    b = vq._read_video_frames(video_b)
    assert a.shape == b.shape == (8, 64, 64, 3)

    mse1, psnr1 = _mse_psnr(a, b)
    mse2, psnr2 = _mse_psnr(a, b)

    # No model, no device — exact equality.
    np.testing.assert_array_equal(mse1, mse2)
    np.testing.assert_array_equal(psnr1, psnr2)

    # And the values are non-trivial — the +5 perturbation gives MSE in the
    # tens, well clear of zero.  (Sanity: catches the case where ffmpeg
    # codec round-trip flattened our delta into nothing.)
    assert mse1.mean() > 1.0, f"MSE collapsed unexpectedly: {mse1!r}"


# ---------------------------------------------------------------------------
# 2) LPIPS — CPU vs CUDA within ULP-level tolerance.
# ---------------------------------------------------------------------------


@_REQUIRES_CUDA
def test_lpips_cpu_cuda_parity(
    synthetic_video_pair: tuple[Path, Path],
    record_property: Callable[[str, object], None],
) -> None:
    """LPIPS(alex) forward should match CPU vs CUDA to within ~1e-4.

    AlexNet runs fp32 on both devices; the only legal drift is cuDNN
    picking different algorithmic paths for conv2d vs CPU MKL.  Item I's
    spec sets the failure threshold at "diff > 1e-4 indicates a real
    numerical issue" — we enforce ``atol=1e-4, rtol=1e-4`` accordingly.
    Observed on RTX 2000 Ada / cuDNN 12.x: ~1.5e-5 absolute, ~3e-3
    relative on the smallest-magnitude entries (rtol-only failures here
    are dominated by ~1e-3 LPIPS values — absolute diff is the load-
    bearing check).  Captured via ``record_property`` so the JUnit XML
    surfaces the observed numbers.
    """
    import numpy as np

    vq = _load_script("verify_quality_parity_mod", _VQ_SRC)
    video_a, video_b = synthetic_video_pair
    a = vq._read_video_frames(video_a)
    b = vq._read_video_frames(video_b)

    lpips_cpu = np.asarray(
        _lpips_per_frame_on_device(a, b, "cpu"), dtype=np.float64
    )
    lpips_cuda = np.asarray(
        _lpips_per_frame_on_device(a, b, "cuda"), dtype=np.float64
    )

    # Sanity: non-trivial distances (not all-zero, not all-saturated).
    assert lpips_cpu.size == lpips_cuda.size == 8
    assert (lpips_cpu > 1e-4).all(), (
        f"LPIPS distances collapsed to ~0; perturbation lost? {lpips_cpu!r}"
    )
    assert (lpips_cpu < 0.95).all(), (
        f"LPIPS saturated near 1.0; perturbation too large? {lpips_cpu!r}"
    )

    abs_diff = np.abs(lpips_cpu - lpips_cuda)
    max_abs_diff = float(abs_diff.max())
    max_rel_diff = float(
        np.max(abs_diff / np.maximum(np.abs(lpips_cpu), 1e-12))
    )
    # Surface observed numbers in JUnit XML for the report.
    record_property("lpips_max_abs_diff", max_abs_diff)
    record_property("lpips_max_rel_diff", max_rel_diff)
    record_property("lpips_cpu_mean", float(lpips_cpu.mean()))
    record_property("lpips_cuda_mean", float(lpips_cuda.mean()))

    # atol=1e-4 is the spec's "real numerical issue" threshold from Item I.
    # rtol is set equally loose so the smallest-magnitude entries (~1e-3
    # LPIPS) don't trip on cuDNN's per-conv algorithm choice; the
    # absolute-diff bound is the load-bearing check for correctness.
    np.testing.assert_allclose(
        lpips_cuda,
        lpips_cpu,
        atol=1e-4,
        rtol=1e-4,
        err_msg=(
            "LPIPS CPU vs CUDA diverged beyond the Item I spec "
            "threshold (atol=1e-4 indicates a real numerical issue): "
            f"max_abs={max_abs_diff:.3e}, max_rel={max_rel_diff:.3e}"
        ),
    )


# ---------------------------------------------------------------------------
# 3) FVD parity — explicitly skipped.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "FVD parity is out of scope here: scripts/compute_fvd.py loads I3D "
        "weights from Facebook's pytorchvideo URL; a cold cache on an "
        "offline VM stalls indefinitely.  See docs/METHODOLOGY.md "
        "'Held-out reference set for FVD' for the supported eval path."
    )
)
def test_fvd_cpu_cuda_parity() -> None:  # pragma: no cover - skip marker
    pass


# ---------------------------------------------------------------------------
# 4) End-to-end sanity — invoke the CLI; confirm a RESULT line on CPU.
# ---------------------------------------------------------------------------


@_REQUIRES_CUDA
def test_eval_cpu_quality_cli_smoke(
    synthetic_video_pair: tuple[Path, Path],
) -> None:
    """Run scripts/eval_cpu_quality.py on the synthetic pair, parse RESULT.

    Sanity check, not parity: the script forces CPU internally regardless
    of host silicon (that's the point of Item F).  We confirm it produces
    a valid RESULT JSON line with the expected pixel sub-dict shape.
    """
    video_a, video_b = synthetic_video_pair
    proc = subprocess.run(
        [sys.executable, str(_EVAL_SRC), str(video_a), str(video_b)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(_HERE.parent),
    )
    assert proc.returncode == 0, (
        f"eval_cpu_quality exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    result_lines = [
        line for line in proc.stdout.splitlines() if line.startswith("RESULT ")
    ]
    assert len(result_lines) == 1, (
        f"expected exactly one RESULT line, got {len(result_lines)}\n"
        f"stdout:\n{proc.stdout}"
    )
    payload = json.loads(result_lines[0][len("RESULT "):])
    assert payload["mode"] == "cpu_quality_eval"
    assert payload["fvd"] is None  # no --fvd-* flags passed
    pixel = payload["pixel"]
    assert pixel["n_frames"] == 8
    assert pixel["lpips_net"] == "alex"
    assert isinstance(pixel["mse_mean"], float)
    assert isinstance(pixel["psnr_mean"], float)
    assert isinstance(pixel["lpips_mean"], float)
    # Confirm the script self-identifies as CPU on stderr.
    assert "Device: cpu" in proc.stderr, (
        f"expected 'Device: cpu' in stderr, got:\n{proc.stderr}"
    )
