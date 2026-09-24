#!/usr/bin/env python3
"""Quantitative quality comparison between two video mp4s.

For each pair of frames at the same index, computes:
  * MSE       — raw pixel-wise mean squared error (uint8 scale, lower is closer)
  * PSNR      — peak signal-to-noise ratio (dB; higher is closer)
  * LPIPS     — learned perceptual image patch similarity (AlexNet backbone)
                  Range [0, 1+] — 0 is identical, ~0.1 is "perceptually similar",
                  ~0.3+ is "noticeably different." Standard reference for
                  diffusion-model quality eval.

Reports per-frame and aggregate (mean, max). Designed for verifying that the
adaptive cache (and / or FP8 wiring) does not degrade quality vs. the
no-cache baseline at the same prompt + seed.

Usage:
    .venv/bin/python scripts/verify_quality.py \
        benchmark-results/cosmos_adaptive_final.mp4 \
        benchmark-results/cosmos_adaptive_fp8_final.mp4

Both videos must have the same frame count and resolution.

LPIPS runs on CUDA if available; otherwise CPU. The eval is single-pass — no
inference, no training; just forward calls over each frame pair.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _read_video_frames(path: Path) -> np.ndarray:
    """Decode the mp4 into an (T, H, W, C) uint8 numpy array via imageio."""
    import imageio.v3 as iio
    import numpy as np

    # `pyav` is the fast path but is an extra; `pyav-mini` / `FFMPEG` work too.
    # imageio-ffmpeg is already a runtime dep (used by run_cosmos.py for mp4
    # write), so the ffmpeg plugin is always available.
    frames = iio.imread(path, plugin="FFMPEG")
    if frames.ndim == 3:  # single frame
        frames = frames[None, ...]
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames


def _to_lpips_tensor(frame_uint8: np.ndarray, device: str) -> object:
    """uint8 HWC -> torch CHW float in [-1, 1] (LPIPS's expected range)."""
    import torch

    f = torch.from_numpy(frame_uint8).to(device, non_blocking=True).float()
    f = f.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0  # (1, 3, H, W) in [-1, 1]
    return f


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_a", type=Path,
                        help="reference video (e.g. no-cache or baseline)")
    parser.add_argument("video_b", type=Path,
                        help="candidate video (e.g. adaptive cache or FP8)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu / mps; default = cuda if available")
    parser.add_argument("--net", default="alex",
                        help="LPIPS backbone: alex | vgg | squeeze (default alex)")
    parser.add_argument("--per-frame", action="store_true",
                        help="print every frame's metrics, not just aggregates")
    args = parser.parse_args()

    import lpips
    import numpy as np
    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    a = _read_video_frames(args.video_a)
    b = _read_video_frames(args.video_b)

    if a.shape != b.shape:
        print(f"ERROR: shape mismatch.  a={a.shape}  b={b.shape}")
        return 2
    n_frames, h, w, c = a.shape
    print(f"Compared: {args.video_a.name} vs {args.video_b.name}")
    print(f"          {n_frames} frames @ {w}x{h}, channels={c}")
    print(f"Device:   {args.device}   LPIPS backbone: {args.net}")

    # MSE / PSNR — vectorized in numpy across all frames at once for speed.
    diff = a.astype(np.int32) - b.astype(np.int32)
    mse_per_frame = (diff.astype(np.float64) ** 2).mean(axis=(1, 2, 3))
    psnr_per_frame = np.where(
        mse_per_frame > 0,
        10.0 * np.log10((255.0 ** 2) / np.where(mse_per_frame > 0, mse_per_frame, 1.0)),
        float("inf"),
    )

    # LPIPS — per-frame forward through the model.
    print("Loading LPIPS network...")
    loss_fn = lpips.LPIPS(net=args.net, verbose=False).to(args.device)
    loss_fn.eval()

    lpips_per_frame: list[float] = []
    with torch.inference_mode():
        for i in range(n_frames):
            ta = _to_lpips_tensor(a[i], args.device)
            tb = _to_lpips_tensor(b[i], args.device)
            d = loss_fn(ta, tb).item()
            lpips_per_frame.append(d)
            if args.per_frame:
                print(f"  frame {i:3d}:  MSE={mse_per_frame[i]:8.3f}  "
                      f"PSNR={psnr_per_frame[i]:6.2f} dB  "
                      f"LPIPS={d:.4f}")

    lpips_arr = np.array(lpips_per_frame, dtype=np.float64)
    print("\n=== Aggregate ===")

    def _row(name: str, arr: np.ndarray, fmt: str = "{:.4f}") -> None:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            print(f"  {name:10s}  all non-finite")
            return
        print(
            f"  {name:10s}  mean=" + fmt.format(finite.mean()) +
            "  median=" + fmt.format(float(np.median(finite))) +
            "  min="    + fmt.format(finite.min()) +
            "  max="    + fmt.format(finite.max())
        )

    _row("MSE",   mse_per_frame, "{:.3f}")
    _row("PSNR",  psnr_per_frame, "{:.2f}")
    _row("LPIPS", lpips_arr,      "{:.4f}")

    # Interpretation hint — LPIPS reference bands from the literature.
    mean_lpips = float(lpips_arr.mean())
    if mean_lpips < 0.05:
        band = "near-identical (well under perceptual threshold)"
    elif mean_lpips < 0.12:
        band = "perceptually very similar"
    elif mean_lpips < 0.25:
        band = "small but visible perceptual differences"
    elif mean_lpips < 0.4:
        band = "noticeably different"
    else:
        band = "substantially different"
    print(f"\n  LPIPS interpretation: {band}")

    # Special-case identical inputs.
    if not math.isfinite(psnr_per_frame.mean()) or (mse_per_frame == 0).all():
        print("\n  NOTE: at least one frame pair is byte-identical (MSE=0). "
              "This is expected for byte-identical mp4s (e.g. seed-0 "
              "determinism with same config).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
