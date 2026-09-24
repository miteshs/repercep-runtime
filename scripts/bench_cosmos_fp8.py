"""Benchmark Cosmos-Predict-7B FP8 attention wiring on MI300X.

This is the head-to-head end-to-end measurement for Phase 2.5: the same
adaptive-caching config run twice, once with the FP8 backend bridge active
(``REPERCEP_FP8_ATTENTION=1``) and once with the default native dispatcher.
F19 showed the env var was a no-op until this session; the bench number
demonstrates the wiring is actually live.

Usage:

    sg render -c "sg video -c '.venv/bin/python scripts/bench_cosmos_fp8.py'"
    .venv/bin/python scripts/bench_cosmos_fp8.py --frames 49 --steps 12

The default config matches the publishable headline (121f / 36 steps /
adaptive thr=0.30 / force_full_every=16). Output mp4s go to
``benchmark-results/bench_cosmos_fp8_*.mp4``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _setup_imports() -> None:
    """Add this worktree's ``src/`` and ``kernels/`` to ``sys.path``.

    The shared ``.venv`` carries an editable install of repercep-runtime; without
    this hook a script invoked from a parallel worktree would import the
    *other* worktree's source. Mirror the pattern from ``scripts/bench_fp8.py``.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for sub in ("src", "kernels"):
        path = repo_root / sub
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_setup_imports()


def main() -> int:
    parser = argparse.ArgumentParser(description="Cosmos FP8 backend head-to-head")
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
    parser.add_argument("--adaptive-threshold", type=float, default=0.3)
    parser.add_argument("--force-full-every", type=int, default=16)
    parser.add_argument("--out-dir", default="benchmark-results")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["adaptive", "adaptive_fp8"],
        choices=("adaptive", "adaptive_fp8"),
        help="subset of configs to run",
    )
    args = parser.parse_args()

    import torch
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    from repercep.attention.diffusers_backend import (
        activate_repercep_fp8_backend,
        register_repercep_fp8_backend,
    )
    from repercep.backend.registry import select_backend
    from repercep.models.cosmos import CosmosConfig, CosmosEngine
    from repercep.runtime.denoise import DenoiseStats, denoise_cosmos_video
    from repercep.runtime.types import GenerationParams, GenerationRequest

    # Make absolutely sure the bridge is registered before any pipeline load
    # (it would be, from the package init — defensive paranoia for the bench).
    register_repercep_fp8_backend()

    backend = select_backend()
    device = backend.devices()[0]
    print(
        f"[bench] backend={backend.name}  device={device.name}  "
        f"{device.total_memory_gib:.0f} GiB",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # The model load is amortized across runs — both configs share the same
    # pipe, scheduler, etc. The bench only changes the active attention
    # backend in diffusers' dispatcher.
    engine = CosmosEngine(
        backend,
        CosmosConfig(
            use_native_loop=True,
            cache_mode="adaptive",
            cache_adaptive_threshold=args.adaptive_threshold,
            cache_force_full_every=args.force_full_every,
        ),
    )
    print("[bench] loading Cosmos-Predict-7B (~38 GB) ...", flush=True)
    t0 = time.perf_counter()
    engine.load()
    print(f"[bench] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    summaries: list[dict[str, Any]] = []
    for label in args.configs:
        # Activate the requested backend for this run.
        if label == "adaptive_fp8":
            os.environ["REPERCEP_FP8_ATTENTION"] = "1"
            activate_repercep_fp8_backend()
            assert (
                _AttentionBackendRegistry._active_backend.value == "repercep_fp8"
            ), "repercep_fp8 not active despite activation request"
        else:
            os.environ.pop("REPERCEP_FP8_ATTENTION", None)
            _AttentionBackendRegistry.set_active_backend(AttentionBackendName.NATIVE)
        active = _AttentionBackendRegistry._active_backend
        print(f"[bench] config={label}  active_backend={active.value}", flush=True)

        stats = DenoiseStats()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
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
            cache_mode=engine._config.cache_mode,
            cache_adaptive_threshold=engine._config.cache_adaptive_threshold,
            cache_force_full_every=engine._config.cache_force_full_every,
            stats=stats,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t1
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3

        frames = _video_to_uint8(video_raw[0])
        n_frames = int(frames.shape[0])
        first_std = float(frames[0].float().std())
        last_std = float(frames[-1].float().std())
        diffs = (frames[1:].float() - frames[:-1].float()).abs().mean()
        mean_motion = float(diffs)

        out_path = out_dir / f"bench_cosmos_fp8_{label}.mp4"
        _save_video(frames, out_path)

        summary = {
            "config": label,
            "active_backend": active.value,
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
        summaries.append(summary)
        print("[bench] RESULT " + json.dumps(summary), flush=True)

    # Relative speedup line — the headline this whole exercise produces.
    if len(summaries) == 2:
        a, b = summaries
        speedup = a["elapsed_s"] / b["elapsed_s"]
        print(
            f"[bench] {b['config']} vs {a['config']}: "
            f"{speedup:.2f}x ({a['elapsed_s']}s -> {b['elapsed_s']}s)",
            flush=True,
        )

    return 0


def _video_to_uint8(video: Any) -> Any:
    """Normalize diffusers video output to (T, H, W, 3) uint8 on CPU.

    Mirrors ``scripts/bench_caching.py``'s helper so the bench is self-
    contained.
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
    """Write frames (T, H, W, 3) uint8 to mp4 via imageio (PNG fallback)."""
    import numpy as np

    arr = frames.numpy() if hasattr(frames, "numpy") else np.asarray(frames)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, arr, fps=fps, codec="libx264")
        return path
    except Exception as exc:
        print(f"[bench] mp4 encode unavailable ({exc}); writing PNG frames", flush=True)
        from PIL import Image

        frame_dir = path.with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(arr):
            Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")
        return frame_dir


if __name__ == "__main__":
    raise SystemExit(main())
