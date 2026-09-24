#!/usr/bin/env python3
"""CPU-only quality eval — LPIPS+MSE+PSNR (single-pair) + optional FVD (set).

A thin convenience runner that chains the two existing quality scripts in this
repo into one CPU-focused workflow:

  1. Single-pair pixel/perceptual metrics — what ``scripts/verify_quality.py``
     reports: MSE, PSNR, and LPIPS (AlexNet backbone) across matched frames.
     Requires the reference and candidate videos to have identical shape
     ``(T, H, W, C)``.

  2. (Optional) Distribution-level FVD — what ``scripts/compute_fvd.py``
     reports: the Fréchet distance between two sets of I3D-extracted feature
     distributions.  This is the metric you want when the cache trades
     trajectory equivalence for compute (subsequent pixel-level metrics on a
     specific (prompt, seed) pair under-report distribution-level fidelity).

This script:
  * Forces ``--device cpu`` on **both** inner stages.  This is the CPU eval
    flow — no GPU acceleration of the evaluator itself, regardless of what the
    surrounding host has.  (The generators that produced the videos may have
    run on any silicon; the eval is CPU.)
  * Emits one combined ``RESULT`` JSON line at the end, plus a human summary
    block.  The JSON layout is:

        {"mode": "cpu_quality_eval",
         "pixel": {"mse_mean": ..., "psnr_mean": ..., "lpips_mean": ...,
                    "n_frames": ..., "lpips_net": "alex"},
         "fvd": {...} | null,
         "reference": "...mp4", "candidate": "...mp4",
         "fvd_reference_set": [...] | null,
         "fvd_candidate_set": [...] | null}

    The ``fvd`` sub-dict (when present) mirrors ``compute_fvd()``'s return
    shape: ``{"mode": "fvd"|"feature_l2", "fvd"|"feature_l2": ...,
    "n_reference_clips": ..., "n_candidate_clips": ..., "feature_dim": ...}``.

  * Forwards the loud small-N FVD warning from ``compute_fvd`` straight to
    the user's stderr (it's load-bearing — see
    ``docs/METHODOLOGY.md`` §"Held-out reference set for FVD").

See also:
  * ``docs/METHODOLOGY.md`` §"Held-out reference set for FVD" — workflow,
    directory layout, and the rationale for N ≥ 50.
  * ``docs/COSMOS_ON_CPU.md`` / ``docs/WAN_ON_CPU.md`` — model-specific CPU
    runners that produce the videos this script consumes.

CLI:
    python scripts/eval_cpu_quality.py REFERENCE.mp4 CANDIDATE.mp4
    python scripts/eval_cpu_quality.py REFERENCE.mp4 CANDIDATE.mp4 \\
        --fvd-reference-set held_out_refs/ \\
        --fvd-candidate-set adaptive_outputs/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

# scripts/ is not a package, so we load compute_fvd and verify_quality by path.
# Mirrors the pattern in tests/test_fvd.py.
_HERE = Path(__file__).resolve().parent


def _load_script_module(name: str, src: Path):
    spec = importlib.util.spec_from_file_location(name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Lazy module loaders — keeps `--help` cheap and lets the tests skip the
# torch/lpips/imageio imports for argument-parsing-only paths.
def _compute_fvd_mod():
    return _load_script_module("compute_fvd_mod", _HERE / "compute_fvd.py")


def _verify_quality_mod():
    return _load_script_module("verify_quality_mod", _HERE / "verify_quality.py")


# Video file extensions we look for inside `--fvd-*-set` directories.
_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")


def _collect_videos(directory: Path) -> list[Path]:
    """Return sorted list of video files inside ``directory`` (non-recursive).

    The held-out reference workflow (METHODOLOGY §"Held-out reference set for
    FVD") expects a flat directory of one mp4 per prompt-seed pair, so we do
    not recurse.  Errors loudly if the directory is empty.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory}: not a directory")
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    )
    if not files:
        raise FileNotFoundError(
            f"{directory}: no video files found "
            f"(looked for {', '.join(_VIDEO_EXTS)})"
        )
    return files


def _compute_pixel_metrics(
    reference: Path,
    candidate: Path,
    *,
    lpips_net: str = "alex",
) -> dict[str, object]:
    """LPIPS+MSE+PSNR on matched-shape ``(reference, candidate)``, on CPU.

    Re-uses the per-frame and tensor-prep helpers from ``verify_quality.py``
    so the math stays identical to the CLI tool.  Returns a dict with the
    aggregated metrics (means over frames) plus the frame count and the
    LPIPS backbone name for traceability in the RESULT line.
    """
    import lpips
    import numpy as np
    import torch

    vq = _verify_quality_mod()

    a = vq._read_video_frames(reference)
    b = vq._read_video_frames(candidate)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: reference={a.shape} candidate={b.shape}"
        )
    n_frames = int(a.shape[0])

    # MSE / PSNR — vectorised across all frames at once (same formula as
    # verify_quality.main's inline computation).
    diff = a.astype(np.int32) - b.astype(np.int32)
    mse_per_frame = (diff.astype(np.float64) ** 2).mean(axis=(1, 2, 3))
    psnr_per_frame = np.where(
        mse_per_frame > 0,
        10.0 * np.log10(
            (255.0 ** 2) / np.where(mse_per_frame > 0, mse_per_frame, 1.0)
        ),
        float("inf"),
    )

    # LPIPS — CPU forward through the AlexNet backbone.
    loss_fn = lpips.LPIPS(net=lpips_net, verbose=False).to("cpu")
    loss_fn.eval()

    lpips_per_frame: list[float] = []
    with torch.inference_mode():
        for i in range(n_frames):
            ta = vq._to_lpips_tensor(a[i], "cpu")
            tb = vq._to_lpips_tensor(b[i], "cpu")
            d = loss_fn(ta, tb).item()
            lpips_per_frame.append(float(d))

    lpips_arr = np.asarray(lpips_per_frame, dtype=np.float64)
    finite_psnr = psnr_per_frame[np.isfinite(psnr_per_frame)]
    psnr_mean = float(finite_psnr.mean()) if finite_psnr.size else float("inf")

    return {
        "mse_mean": float(mse_per_frame.mean()),
        "mse_max": float(mse_per_frame.max()),
        "psnr_mean": psnr_mean,
        "psnr_min": float(finite_psnr.min()) if finite_psnr.size else float("inf"),
        "lpips_mean": float(lpips_arr.mean()) if lpips_arr.size else 0.0,
        "lpips_max": float(lpips_arr.max()) if lpips_arr.size else 0.0,
        "n_frames": n_frames,
        "lpips_net": lpips_net,
    }


def _compute_distribution_fvd(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    num_clips: int,
    n_warn: int,
) -> dict[str, object]:
    """Run ``compute_fvd.compute_fvd`` over flat directories of videos.

    Returns the raw dict from ``compute_fvd()`` with two extra keys for
    JSON traceability: the sorted lists of file paths used for each side.
    """
    fvd_mod = _compute_fvd_mod()
    refs = _collect_videos(reference_dir)
    cands = _collect_videos(candidate_dir)
    result = fvd_mod.compute_fvd(
        refs,
        cands,
        device="cpu",
        num_clips=num_clips,
        n_warn=n_warn,
    )
    return {
        **result,
        "reference_paths": [str(p) for p in refs],
        "candidate_paths": [str(p) for p in cands],
    }


def _human_summary(combined: dict[str, object]) -> str:
    """Render the same numbers as the JSON, but for the human reader."""
    lines: list[str] = []
    lines.append("=== CPU Quality Eval ===")
    pix = combined["pixel"]
    assert isinstance(pix, dict)
    lines.append(f"  reference:        {combined['reference']}")
    lines.append(f"  candidate:        {combined['candidate']}")
    lines.append(f"  frames compared:  {pix['n_frames']}")
    lines.append(f"  LPIPS backbone:   {pix['lpips_net']}")
    lines.append("")
    lines.append("--- Pixel/perceptual (single-pair) ---")
    lines.append(f"  MSE   mean={pix['mse_mean']:.3f}   max={pix['mse_max']:.3f}")
    psnr_mean = pix["psnr_mean"]
    if isinstance(psnr_mean, float) and not math.isfinite(psnr_mean):
        lines.append("  PSNR  mean=inf   (byte-identical frames)")
    else:
        lines.append(
            f"  PSNR  mean={pix['psnr_mean']:.2f} dB   min={pix['psnr_min']:.2f} dB"
        )
    lines.append(
        f"  LPIPS mean={pix['lpips_mean']:.4f}   max={pix['lpips_max']:.4f}"
    )

    fvd = combined["fvd"]
    if fvd is None:
        lines.append("")
        lines.append("--- Distribution FVD: not requested ---")
        lines.append(
            "  (pass --fvd-reference-set DIR --fvd-candidate-set DIR to run)"
        )
    else:
        assert isinstance(fvd, dict)
        lines.append("")
        lines.append("--- Distribution FVD (set-level) ---")
        lines.append(f"  reference clips:  {fvd['n_reference_clips']}")
        lines.append(f"  candidate clips:  {fvd['n_candidate_clips']}")
        lines.append(f"  feature dim:      {fvd['feature_dim']}")
        if fvd["mode"] == "fvd":
            lines.append(f"  FVD:              {fvd['fvd']:.4f}")
        else:
            lines.append(f"  feature L2:       {fvd['feature_l2']:.4f}")
            lines.append(
                "  (N too small for FVD; reporting per-clip feature L2 fallback)"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser — split out so tests can introspect args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="reference video (e.g. no-cache baseline mp4)",
    )
    parser.add_argument(
        "candidate",
        type=Path,
        help="candidate video (e.g. adaptive cache mp4)",
    )
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=("alex", "vgg", "squeeze"),
        help="LPIPS backbone (default: alex — same as verify_quality.py)",
    )
    parser.add_argument(
        "--fvd-reference-set",
        type=Path,
        default=None,
        help=(
            "directory of reference-set videos for distribution-level FVD; "
            "see docs/METHODOLOGY.md §'Held-out reference set for FVD'"
        ),
    )
    parser.add_argument(
        "--fvd-candidate-set",
        type=Path,
        default=None,
        help=(
            "directory of candidate-set videos for distribution-level FVD; "
            "must share prompt+seed pairing with --fvd-reference-set"
        ),
    )
    parser.add_argument(
        "--fvd-num-clips",
        type=int,
        default=1,
        help="non-overlapping 8-frame clips per video for FVD (default: 1)",
    )
    parser.add_argument(
        "--fvd-n-warn",
        type=int,
        default=50,
        help=(
            "emit the loud small-N FVD warning when min(N_ref, N_cand) is "
            "below this (default: 50 — see METHODOLOGY rationale)"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Programmatic entrypoint — runs both stages and returns the combined dict."""
    if (args.fvd_reference_set is None) != (args.fvd_candidate_set is None):
        raise ValueError(
            "--fvd-reference-set and --fvd-candidate-set must be provided "
            "together (or neither)"
        )
    if not args.reference.exists():
        raise FileNotFoundError(args.reference)
    if not args.candidate.exists():
        raise FileNotFoundError(args.candidate)

    pixel = _compute_pixel_metrics(
        args.reference,
        args.candidate,
        lpips_net=args.lpips_net,
    )

    fvd: dict[str, object] | None = None
    if args.fvd_reference_set is not None:
        assert args.fvd_candidate_set is not None
        fvd = _compute_distribution_fvd(
            args.fvd_reference_set,
            args.fvd_candidate_set,
            num_clips=args.fvd_num_clips,
            n_warn=args.fvd_n_warn,
        )

    combined: dict[str, object] = {
        "mode": "cpu_quality_eval",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "pixel": pixel,
        "fvd": fvd,
        "fvd_reference_set": (
            str(args.fvd_reference_set) if args.fvd_reference_set else None
        ),
        "fvd_candidate_set": (
            str(args.fvd_candidate_set) if args.fvd_candidate_set else None
        ),
    }
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print(
        f"CPU quality eval: {args.reference.name} vs {args.candidate.name}",
        file=sys.stderr,
    )
    print("Device: cpu (forced — this is the CPU eval flow)", file=sys.stderr)

    combined = run(args)

    print()
    print(_human_summary(combined))
    print()
    # Single-line RESULT for downstream parsing (mirrors run_cosmos.py /
    # run_wan.py convention).
    print(f"RESULT {json.dumps(combined, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
