#!/usr/bin/env python3
"""Benchmark Cosmos-Predict-7B caching modes head-to-head on MI300X.

Runs the same 121f / 36-step config under three caching configurations and
prints wall-time + full-forward counts + simple visual-quality sanity stats
(frame count + first/last frame std + inter-frame motion). The point is to
verify the adaptive (TeaCache-style) gate matches the fixed-cadence
``cache_skip_every=4`` baseline within 10% wall time while triggering at most
as many full forwards (and ideally fewer or at quality parity).

Usage:

    sg render -c "sg video -c '.venv/bin/python scripts/bench_caching.py'"
    .venv/bin/python scripts/bench_caching.py --frames 121 --steps 36 \\
        --adaptive-threshold 0.1

Outputs (gitignored): ``benchmark-results/bench_caching_<config>.mp4`` for each
configuration, plus a JSON summary line per run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Cosmos caching modes")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default=(
            "A sleek autonomous delivery robot rolls along a sunlit city "
            "sidewalk past glass storefronts, smooth forward motion, "
            "photorealistic, high detail."
        ),
    )
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=0.3,
        help=(
            "adaptive cache: accumulated rel-L1 threshold for triggering a full forward "
            "(tuned default 0.3 on 121f/36step)"
        ),
    )
    parser.add_argument(
        "--force-full-every",
        type=int,
        default=16,
        help="adaptive cache: force a full forward at least every N steps",
    )
    parser.add_argument(
        "--fixed-skip-every",
        type=int,
        default=4,
        help="fixed cache: full forward every Nth step (matches headline F16+F17 cache=4)",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark-results",
        help="directory for output mp4s and JSON summaries",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["none", "fixed", "adaptive"],
        choices=("none", "fixed", "adaptive"),
        help="subset of configs to run",
    )
    args = parser.parse_args()

    from repercep.backend.registry import select_backend
    from repercep.models.cosmos import CosmosConfig, CosmosEngine
    from repercep.runtime.denoise import DenoiseStats, denoise_cosmos_video
    from repercep.runtime.types import GenerationParams, GenerationRequest

    backend = select_backend()
    device = backend.devices()[0]
    print(
        f"[bench] backend={backend.name}  device={device.name}  "
        f"{device.total_memory_gib:.0f} GiB",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configs to benchmark. Each is (label, CosmosConfig kwargs, denoise kwargs)
    plans: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    if "none" in args.configs:
        plans.append(("none", {"use_native_loop": True}, {}))
    if "fixed" in args.configs:
        plans.append(
            (
                f"fixed_skip{args.fixed_skip_every}",
                {
                    "use_native_loop": True,
                    "cache_skip_every": args.fixed_skip_every,
                    "cache_mode": "fixed",
                },
                {},
            )
        )
    if "adaptive" in args.configs:
        plans.append(
            (
                f"adaptive_thr{args.adaptive_threshold:.3f}",
                {
                    "use_native_loop": True,
                    "cache_mode": "adaptive",
                    "cache_adaptive_threshold": args.adaptive_threshold,
                    "cache_force_full_every": args.force_full_every,
                },
                {},
            )
        )

    request = GenerationRequest(
        prompt=args.prompt,
        params=GenerationParams(
            num_frames=args.frames,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance,
            seed=args.seed,
        ),
    )

    summaries: list[dict[str, Any]] = []
    # Load the model once; reuse the pipe across runs. The bench is about the
    # caching loop, not load time.
    engine: CosmosEngine | None = None
    for label, cfg_kwargs, _ in plans:
        if engine is None:
            engine = CosmosEngine(backend, CosmosConfig(**cfg_kwargs))
            engine.load()
        else:
            # Update config in place; the pipe is reusable.
            for key, value in cfg_kwargs.items():
                setattr(engine._config, key, value)

        # Run the native loop directly so we can collect DenoiseStats.
        stats = DenoiseStats()
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        video_raw = denoise_cosmos_video(
            engine.pipeline,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.params.height,
            width=request.params.width,
            num_frames=request.params.num_frames,
            num_inference_steps=request.params.num_inference_steps,
            guidance_scale=request.params.guidance_scale,
            fps=request.params.fps,
            seed=request.params.seed,
            output_type="pt",
            cache_skip_every=engine._config.cache_skip_every,
            cache_warmup_steps=engine._config.cache_warmup_steps,
            cache_mode=engine._config.cache_mode,
            cache_adaptive_threshold=engine._config.cache_adaptive_threshold,
            cache_force_full_every=engine._config.cache_force_full_every,
            stats=stats,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3

        # Quality sanity: frame count, first/last frame std, mean inter-frame
        # motion. We can't quality-check without eyeballing, but these stats
        # bracket "did the loop produce something pathologically static".
        frames = _video_to_uint8(video_raw[0])
        n_frames = int(frames.shape[0])
        first_std = float(frames[0].float().std())
        last_std = float(frames[-1].float().std())
        # Mean inter-frame absolute diff in [0, 255].
        diffs = (frames[1:].float() - frames[:-1].float()).abs().mean()
        mean_motion = float(diffs)

        out_path = out_dir / f"bench_caching_{label}.mp4"
        _save_video(frames, out_path)

        summary = {
            "config": label,
            "frames": args.frames,
            "steps": args.steps,
            "elapsed_s": round(elapsed, 2),
            "full_forwards": stats.full_forwards,
            "skipped": stats.skipped,
            "peak_hbm_gib": round(peak_gib, 1),
            "n_frames_out": n_frames,
            "first_frame_std": round(first_std, 2),
            "last_frame_std": round(last_std, 2),
            "mean_inter_frame_motion": round(mean_motion, 2),
            "video": str(out_path),
        }
        if engine._config.cache_mode == "adaptive":
            summary["adaptive_threshold"] = engine._config.cache_adaptive_threshold
            summary["force_full_every"] = engine._config.cache_force_full_every
            # First few rel_l1 values to eyeball the gate.
            summary["rel_l1_first8"] = [round(v, 4) for v in stats.rel_l1_history[:8]]
        elif engine._config.cache_mode == "fixed":
            summary["cache_skip_every"] = engine._config.cache_skip_every
        summaries.append(summary)
        print("[bench] RESULT " + json.dumps(summary), flush=True)

    if len(summaries) >= 2:
        # Reference for relative comparison: prefer "fixed" if present, else
        # "none".
        ref = next(
            (s for s in summaries if s["config"].startswith("fixed")),
            summaries[0],
        )
        print(
            f"[bench] reference for relative comparison: {ref['config']}",
            flush=True,
        )
        for summary in summaries:
            if summary["config"] == ref["config"]:
                continue
            wall_ratio = summary["elapsed_s"] / ref["elapsed_s"]
            forwards_delta = summary["full_forwards"] - ref["full_forwards"]
            print(
                f"[bench] vs {ref['config']}: {summary['config']} "
                f"{wall_ratio * 100:.0f}% wall ({summary['elapsed_s']}s vs "
                f"{ref['elapsed_s']}s), forwards Δ={forwards_delta:+d} "
                f"({summary['full_forwards']} vs {ref['full_forwards']})",
                flush=True,
            )

    return 0


def _video_to_uint8(video: Any) -> Any:
    """Normalize a diffusers video output to ``(T, H, W, 3)`` uint8 on CPU.

    Mirrors ``repercep.models.cosmos._as_frame_tensor`` so the bench is
    self-contained.
    """
    import torch

    tensor = video if isinstance(video, torch.Tensor) else torch.as_tensor(video)
    tensor = tensor.detach().to("cpu", dtype=torch.float32)
    if tensor.ndim != 4:
        raise ValueError(f"unexpected video tensor rank: {tuple(tensor.shape)}")
    if tensor.shape[1] in (1, 3):
        tensor = tensor.permute(0, 2, 3, 1)
    return (tensor.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()


def _save_video(frames: Any, path: Path, fps: int = 24) -> Path:
    """Write frames (T, H, W, 3) uint8 to mp4 via imageio."""
    import numpy as np

    arr = frames.numpy() if hasattr(frames, "numpy") else np.asarray(frames)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, arr, fps=fps, codec="libx264")
        return path
    except Exception as exc:
        print(f"[bench] mp4 encode unavailable ({exc}); writing PNGs", flush=True)
        from PIL import Image

        frame_dir = path.with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(arr):
            Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")
        return frame_dir


if __name__ == "__main__":
    raise SystemExit(main())
