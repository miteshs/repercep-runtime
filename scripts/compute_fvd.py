#!/usr/bin/env python3
"""Compute Fréchet Video Distance (FVD) between two sets of videos.

Slots alongside `scripts/verify_quality.py` (LPIPS/MSE/PSNR) as the
distribution-level cache-quality arbiter — the right metric for diffusion
outputs that vary by trajectory while preserving distribution-level fidelity.
See `docs/METHODOLOGY.md` §"Reproducibility envelope" for the broader
framing.

I3D feature extractor (the canonical FVD backbone):
  * Source: ``torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50",
    pretrained=True)``
  * Weights: ``I3D_8x8_R50.pyth`` from
    https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/
    (Kinetics-400, 73.27 % top-1; the standard reference checkpoint).
  * Native clip shape: ``(B, C=3, T=8, H=224, W=224)``.  Kinetics-400
    normalization: mean ``(0.45, 0.45, 0.45)``, std ``(0.225, 0.225, 0.225)``
    per pytorchvideo's ``transforms_factory``.
  * Feature vector: 2048-D, taken at the output of the head's pool + dropout
    (i.e. the input to the classification projection).  This is the standard
    FVD feature tap and matches the original Heusel-style FID adaptation.

Sampling strategy:
  Each video is read into a ``(T, H, W, 3)`` uint8 array via imageio (FFMPEG
  plugin, already on the runtime path).  For a 121-frame Cosmos mp4 we sample
  ``num_clips`` non-overlapping clips of 8 evenly-spaced frames each, mirroring
  the standard FVD literature's multi-clip averaging.  Default
  ``--num-clips=1`` (single centred 8-frame clip across the whole video) for
  parity with our small-N regime; bump it for longer videos.
  Frames are center-cropped to a square, resized to 224x224, scaled to
  ``[0, 1]``, then Kinetics-normalized.

Fréchet distance:
  FVD(A, B) = ||mu_A - mu_B||^2 + tr(S_A + S_B - 2 (S_A S_B)^(1/2))
  ``S`` (sigma) is the empirical covariance over the per-clip feature vectors
  of each set, computed in float64 to keep ``scipy.linalg.sqrtm`` stable.

Small-N caveat (load-bearing — do not strip this):
  FVD literature uses N ≥ 1000.  Our typical cache-quality comparisons have
  N ≤ 10 (often N = 1 head-to-head).  With ``N < 50`` the script prints a
  loud warning; with ``N < 2`` on either side it falls back to a per-clip
  feature L2 distance (since covariance over a single sample is undefined).
  Treat all reported FVDs at small N as **preliminary** indicators — useful
  for relative ordering between candidates, not for absolute comparison
  against published literature values.

CLI:
  .venv/bin/python scripts/compute_fvd.py \\
      --reference benchmark-results/no_cache_*.mp4 \\
      --candidates benchmark-results/adaptive_*.mp4 \\
      [--device cpu|cuda] [--num-clips N] [--n-warn THRESHOLD]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch import nn


# Kinetics-400 normalization per pytorchvideo (transforms_factory.py defaults).
_KINETICS_MEAN = (0.45, 0.45, 0.45)
_KINETICS_STD = (0.225, 0.225, 0.225)

# I3D 8x8 R50 native input shape.
_I3D_FRAMES_PER_CLIP = 8
_I3D_SPATIAL = 224

# Above this set size we suppress the small-N warning.  Default 50 — well below
# the literature norm of 1000 but enough that empirical covariance is at least
# rank-meaningful in 2048-D.
_DEFAULT_N_WARN = 50


def _load_i3d_extractor(device: str) -> nn.Module:
    """Return a callable that maps ``(B, 3, 8, 224, 224)`` → ``(B, 2048)``.

    Implemented by loading the pretrained ``i3d_r50`` and wrapping a stop
    after the head's pool + dropout (the standard FVD feature tap).
    """
    import torch
    from torch import nn

    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "i3d_r50",
        pretrained=True,
        verbose=False,
    )
    model.eval()

    class _I3DFeatureExtractor(nn.Module):
        """Forward through I3D up to the pre-classification pooled feature."""

        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            # blocks[0..5] are the trunk; blocks[6] is the head.
            self.trunk = base.blocks[:6]
            self.head_pool = base.blocks[6].pool
            self.head_dropout = base.blocks[6].dropout

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for block in self.trunk:
                h = block(h)
            h = self.head_pool(h)
            h = self.head_dropout(h)
            # (B, 2048, 1, 1, 1) -> (B, 2048)
            return h.flatten(1)

    extractor = _I3DFeatureExtractor(model).to(device).eval()
    return extractor


def _read_video_frames(path: Path) -> np.ndarray:
    """Decode ``path`` into ``(T, H, W, 3)`` uint8 via imageio's FFMPEG plugin."""
    import imageio.v3 as iio
    import numpy as np

    frames = iio.imread(path, plugin="FFMPEG")
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    if frames.shape[-1] != 3:
        raise ValueError(
            f"{path}: expected 3 channels (RGB), got shape {frames.shape}"
        )
    return frames


def _center_crop_and_resize(frame_uint8: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` uint8 → ``(224, 224, 3)`` uint8 — center crop then resize."""
    import numpy as np
    from PIL import Image

    h, w, _ = frame_uint8.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame_uint8[top : top + side, left : left + side, :]
    img = Image.fromarray(crop)
    img = img.resize((_I3D_SPATIAL, _I3D_SPATIAL), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _video_to_clips(frames: np.ndarray, num_clips: int) -> np.ndarray:
    """Sample ``num_clips`` 8-frame clips of evenly-spaced frames.

    Returns float32 array shape ``(num_clips, 3, 8, 224, 224)``, Kinetics-
    normalized, ready for I3D ingestion.  Center-crops to square then resizes
    to 224 before scaling to [0, 1].
    """
    import numpy as np

    t = frames.shape[0]
    if t < _I3D_FRAMES_PER_CLIP:
        raise ValueError(
            f"video has only {t} frames, need >= {_I3D_FRAMES_PER_CLIP}"
        )

    # Spread `num_clips * _I3D_FRAMES_PER_CLIP` evenly across the full video.
    # For num_clips=1 this is just 8 evenly-spaced frames; for num_clips=k it's
    # k * 8 evenly-spaced frames concatenated, then split into k clips of 8.
    total_target = num_clips * _I3D_FRAMES_PER_CLIP
    idxs = np.linspace(0, t - 1, total_target).round().astype(np.int64)

    clips = np.empty(
        (num_clips, _I3D_FRAMES_PER_CLIP, _I3D_SPATIAL, _I3D_SPATIAL, 3),
        dtype=np.uint8,
    )
    for c in range(num_clips):
        for f in range(_I3D_FRAMES_PER_CLIP):
            global_idx = idxs[c * _I3D_FRAMES_PER_CLIP + f]
            clips[c, f] = _center_crop_and_resize(frames[global_idx])

    # uint8 (N, T, H, W, C) -> float32 (N, C, T, H, W), Kinetics-normalized.
    clips_f = clips.astype(np.float32) / 255.0
    clips_f = np.transpose(clips_f, (0, 4, 1, 2, 3))  # (N, 3, 8, 224, 224)

    mean = np.array(_KINETICS_MEAN, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    std = np.array(_KINETICS_STD, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    clips_f = (clips_f - mean) / std
    return clips_f


def _extract_features(
    paths: list[Path],
    extractor: nn.Module,
    device: str,
    num_clips: int,
) -> np.ndarray:
    """Return ``(N_total_clips, 2048)`` float64 features over all input videos.

    Each video contributes ``num_clips`` clips; total ``len(paths) * num_clips``
    feature vectors.  Features are L2-finite floats in fp64 for downstream
    covariance + sqrtm.
    """
    import numpy as np
    import torch

    feats_all: list[np.ndarray] = []
    for path in paths:
        frames = _read_video_frames(path)
        clips_np = _video_to_clips(frames, num_clips)
        clips = torch.from_numpy(clips_np).to(device)
        with torch.inference_mode():
            f = extractor(clips)
        feats_all.append(f.detach().to("cpu", dtype=torch.float64).numpy())
    return np.concatenate(feats_all, axis=0)


def _frechet_distance(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    sqrtm_eps: float = 1e-6,
) -> float:
    """Standard Fréchet distance between two sets of feature vectors.

    Both arrays are ``(N_i, D)`` in float64.  Returns a scalar float64.
    Cast everything to fp64 before ``sqrtm`` (single-precision sqrtm has
    known stability issues for near-singular matrices).
    """
    import numpy as np
    from scipy import linalg

    feats_a = feats_a.astype(np.float64, copy=False)
    feats_b = feats_b.astype(np.float64, copy=False)

    mu_a = feats_a.mean(axis=0)
    mu_b = feats_b.mean(axis=0)
    # rowvar=False so each row is an observation; D x D output.
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    diff = mu_a - mu_b
    mean_term = float(diff @ diff)

    # sqrtm(Σ_a Σ_b).  Add a tiny diagonal jitter — pure numerical hygiene;
    # would not affect the answer when Σ has decent rank, but stabilises the
    # rank-deficient small-N case.
    d = sigma_a.shape[0]
    eye_jit = np.eye(d, dtype=np.float64) * sqrtm_eps
    # scipy >= 1.16 deprecated `disp=False`; modern call returns just the matrix.
    sqrt_result = linalg.sqrtm((sigma_a + eye_jit) @ (sigma_b + eye_jit))
    # On older scipy a 2-tuple ``(matrix, errest)`` came back; on >= 1.16 it's
    # just the matrix.  Handle both for cross-version safety.
    covmean = sqrt_result[0] if isinstance(sqrt_result, tuple) else sqrt_result
    # sqrtm can return a complex matrix when input is near-PSD with tiny
    # negative imaginary parts; the standard FID convention is to take the
    # real part with an imaginary-component sanity check.
    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0, atol=1e-3):
            max_imag = float(np.abs(covmean.imag).max())
            print(
                f"  WARNING: sqrtm returned non-trivial imaginary part "
                f"(max |Im| = {max_imag:.2e}); using real part. "
                f"This is typical for small-N FVD.",
                file=sys.stderr,
            )
        covmean = covmean.real

    tr_term = float(np.trace(sigma_a + sigma_b - 2.0 * covmean))
    return mean_term + tr_term


def _per_clip_l2_fallback(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """For N=1 vs N=1 we cannot compute covariance; report per-clip L2 instead.

    When ``feats_a`` and ``feats_b`` each have one row, returns the L2 distance
    between their single feature vectors.  When either has more than one row
    but the other has one, returns the mean L2 from the singleton to each
    vector in the larger set.
    """
    import numpy as np

    feats_a = feats_a.astype(np.float64, copy=False)
    feats_b = feats_b.astype(np.float64, copy=False)

    # Mean-feature-to-mean-feature L2 (same as Σ-less Fréchet's mean term).
    mu_a = feats_a.mean(axis=0)
    mu_b = feats_b.mean(axis=0)
    diff = mu_a - mu_b
    return float(np.linalg.norm(diff))


def compute_fvd(
    reference_paths: list[Path],
    candidate_paths: list[Path],
    *,
    device: str,
    num_clips: int = 1,
    n_warn: int = _DEFAULT_N_WARN,
) -> dict[str, object]:
    """High-level entry point used by the CLI and by tests.

    Returns a dict containing ``fvd`` (or ``feature_l2`` if N was too small),
    ``n_reference_clips``, ``n_candidate_clips``, ``feature_dim``, and
    ``mode`` (``"fvd"`` or ``"feature_l2"``).
    """
    if not reference_paths:
        raise ValueError("--reference: no paths supplied")
    if not candidate_paths:
        raise ValueError("--candidates: no paths supplied")
    for p in [*reference_paths, *candidate_paths]:
        if not p.exists():
            raise FileNotFoundError(p)

    extractor = _load_i3d_extractor(device)
    feats_a = _extract_features(reference_paths, extractor, device, num_clips)
    feats_b = _extract_features(candidate_paths, extractor, device, num_clips)

    n_a = feats_a.shape[0]
    n_b = feats_b.shape[0]
    d = feats_a.shape[1]
    assert d == feats_b.shape[1], (
        f"feature-dim mismatch (reference {d} vs candidates {feats_b.shape[1]})"
    )

    # Small-N warning band.
    min_n = min(n_a, n_b)
    if min_n < 2:
        # FVD requires covariance; with a single clip per side it's undefined.
        print(
            f"  NOTE: min(N_ref, N_cand) = {min_n} is too small for FVD; "
            f"reporting per-clip feature L2 distance instead.  "
            f"(FVD literature uses N >= 1000; we have N = {min_n}.)",
            file=sys.stderr,
        )
        l2 = _per_clip_l2_fallback(feats_a, feats_b)
        return {
            "mode": "feature_l2",
            "feature_l2": l2,
            "n_reference_clips": n_a,
            "n_candidate_clips": n_b,
            "feature_dim": d,
        }
    if min_n < n_warn:
        bar = "*" * 60
        print(f"\n{bar}", file=sys.stderr)
        print(
            f"  LOUD WARNING: min(N_ref, N_cand) = {min_n} is well below the "
            f"FVD literature standard of N >= 1000 (we suggest N >= {n_warn}).\n"
            f"  The reported FVD is a preliminary indicator only — useful for\n"
            f"  relative ordering, not absolute comparison against published\n"
            f"  values.  See docs/METHODOLOGY.md §'Reproducibility envelope'.",
            file=sys.stderr,
        )
        print(f"{bar}\n", file=sys.stderr)

    fvd = _frechet_distance(feats_a, feats_b)
    return {
        "mode": "fvd",
        "fvd": fvd,
        "n_reference_clips": n_a,
        "n_candidate_clips": n_b,
        "feature_dim": d,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        nargs="+",
        required=True,
        help="paths to reference videos (the held-out ground-truth set)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        required=True,
        help="paths to candidate videos (what we're scoring)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cpu | mps; default = cuda if available else cpu",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=1,
        help="non-overlapping 8-frame clips to sample per video (default: 1)",
    )
    parser.add_argument(
        "--n-warn",
        type=int,
        default=_DEFAULT_N_WARN,
        help=(
            f"emit a loud small-N warning when min(N_ref, N_cand) < this "
            f"threshold (default: {_DEFAULT_N_WARN})"
        ),
    )
    args = parser.parse_args()

    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Computing FVD: {len(args.reference)} reference videos vs "
        f"{len(args.candidates)} candidate videos  "
        f"({args.num_clips} clip(s) per video)",
        file=sys.stderr,
    )
    print(f"Device: {args.device}", file=sys.stderr)

    result = compute_fvd(
        args.reference,
        args.candidates,
        device=args.device,
        num_clips=args.num_clips,
        n_warn=args.n_warn,
    )

    print()
    print("=== FVD Result ===")
    print(f"  reference clips:  {result['n_reference_clips']}")
    print(f"  candidate clips:  {result['n_candidate_clips']}")
    print(f"  feature dim:      {result['feature_dim']}")
    if result["mode"] == "fvd":
        fvd_val = result["fvd"]
        assert isinstance(fvd_val, float)
        print(f"  FVD:              {fvd_val:.4f}")
        if not math.isfinite(fvd_val):
            print("  WARNING: FVD is not finite — check inputs.")
            return 2
    else:
        l2_val = result["feature_l2"]
        assert isinstance(l2_val, float)
        print(f"  feature L2:       {l2_val:.4f}")
        print(
            "  (N too small for FVD; reporting per-clip feature L2 fallback)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
