#!/usr/bin/env python3
"""Run one Wan-2.2 T2V generation on the Repercep runtime (MI300X).

This is the Wan analogue of ``scripts/run_cosmos.py`` — it loads the Wan-2.2
MoE A14B text-to-video pipeline through Repercep's :class:`WanEngine`, generates
a clip, writes an mp4, and prints a JSON RESULT line.

    .venv/bin/python scripts/run_wan.py --frames 17 --steps 8       # fast smoke
    .venv/bin/python scripts/run_wan.py --frames 81 --steps 40      # reference

The default Wan-2.2 T2V-A14B reference config is 81 frames at 1280x720 with 40
inference steps (5 s at 16 FPS). The smaller TI2V-5B variant is available via
``--small`` for VRAM-constrained smoke tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Wan-2.2 T2V on Repercep / MI300X")
    parser.add_argument(
        "--prompt",
        default=(
            "A sleek autonomous delivery robot rolls along a sunlit city "
            "sidewalk past glass storefronts, smooth forward motion, "
            "photorealistic, high detail."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "blurry, low quality, distorted, overexposed, static, "
            "subtitles, watermark, deformed limbs"
        ),
    )
    # Wan-2.2 T2V-A14B reference defaults: 81 frames at 1280x720, 40 steps.
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument(
        "--guidance-2",
        type=float,
        default=3.0,
        help="MoE second-stage guidance scale (A14B only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark-results/wan_sample.mp4")
    parser.add_argument(
        "--small",
        action="store_true",
        help="use the Wan-2.2 TI2V-5B variant (~10 GiB BF16) instead of T2V-A14B",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the DiT transformer(s) — slow first run, faster steady-state",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help=(
            "decode the VAE in spatial tiles — drops peak VRAM by ~6-8 GiB at "
            "1280x720 so the A14B both-experts-resident path fits in 80 GiB H100"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("rocm", "cuda", "cpu", "auto"),
        default="auto",
        help="compute backend. 'auto' picks the first available (GPU wins on a GPU host).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "wrap the primary run in a per-stage probe (text encode / DiT "
            "high-noise / DiT low-noise / VAE decode / other). Adds forward "
            "hooks + CUDA-syncs around each stage, all numbers are real device "
            "time, no second pass — the printed RESULT and PROFILE lines both "
            "describe the same generation"
        ),
    )
    parser.add_argument(
        "--profile-second-pass",
        action="store_true",
        help=(
            "additionally run a fully-separate profiled generation AFTER the "
            "primary one, with one warmup pass absorbed. Doubles the wall "
            "time of the script. Off by default — the inline --profile is "
            "sufficient for the steady-state breakdown"
        ),
    )
    args = parser.parse_args()

    import torch

    from repercep.backend.registry import select_backend
    from repercep.bench.profile import _Probe, profile_wan
    from repercep.hardware import Vendor
    from repercep.models.wan import NATIVE_FPS, SMALL_REPO, WanConfig, WanEngine
    from repercep.runtime.types import GenerationParams, GenerationRequest

    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    device = backend.devices()[0]
    print(
        f"[repercep] backend={backend.name}  device={device.name}  "
        f"{device.total_memory_gib:.0f} GiB  arch={device.arch}",
        flush=True,
    )

    # --vae-tiling exists to drop GPU HBM peak (~18 GiB on H100 at 1280x720) by
    # decoding the VAE in spatial tiles. On CPU the same flag is actively
    # harmful: it shreds the FP32 VAE decode into hundreds of small per-tile
    # forwards, none of which fit oneDNN's preferred AMX/AVX512 block sizes.
    # Session 17's TI2V-5B 17f/8 smoke ran with --vae-tiling for "parity" with
    # the H100 invocation and cost ~30 min vs the estimated 38-45 min envelope
    # (measured 66 min). Refuse rather than silently flip the flag — the user
    # is making a methodology mistake and silently flipping it hides that from
    # logs.
    if args.vae_tiling and backend.vendor is Vendor.INTEL:
        print(
            "[repercep] FATAL: --vae-tiling is not supported on the CPU backend.\n"
            "  --vae-tiling shreds the FP32 VAE decode on CPU; observed +30 min\n"
            "  on the TI2V-5B 17f/8 smoke. See docs/WAN_ON_CPU.md \xa7\"Methodology\".\n"
            "  Re-run without --vae-tiling.",
            flush=True,
        )
        return 2

    config = WanConfig(
        guidance_scale_2=args.guidance_2,
        compile_transformer=args.compile,
        vae_tiling=args.vae_tiling,
    )
    if args.small:
        config.repo_id = SMALL_REPO
    engine = WanEngine(backend, config)

    size_hint = "~10 GiB BF16" if args.small else "~52 GiB BF16"
    print(f"[repercep] loading {config.repo_id} ({size_hint}) ...", flush=True)
    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0
    print(f"[repercep] model loaded in {load_s:.1f}s", flush=True)

    request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        params=GenerationParams(
            num_frames=args.frames,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance,
            fps=NATIVE_FPS,  # Wan was trained at 16 FPS.
            seed=args.seed,
        ),
    )
    print(
        f"[repercep] generating {args.frames} frames @ {args.width}x{args.height}, "
        f"{args.steps} steps, seed {args.seed} ...",
        flush=True,
    )
    cpu_run = backend.vendor is Vendor.INTEL
    rss_before = 0
    if cpu_run:
        import resource
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    else:
        torch.cuda.reset_peak_memory_stats()

    # Inline per-stage probe — forward hooks fire outside any compiled graph,
    # so the breakdown is valid whether or not the DiT is torch.compile'd.
    probe: _Probe | None = None
    if args.profile:
        probe = _Probe(engine.pipeline)
        probe.__enter__()

    t1 = time.perf_counter()
    frames = list(engine.generate(request))
    gen_s = time.perf_counter() - t1

    prof_payload: dict[str, object] | None = None
    if probe is not None:
        probe.__exit__()
        dit_hi = probe.dit.seconds
        dit_lo = probe.dit2.seconds
        prof_payload = {
            "total_s": round(gen_s, 2),
            "text_encode_s": round(probe.text.seconds, 3),
            "dit_loop_s": round(dit_hi + dit_lo, 2),
            "dit_high_noise_s": round(dit_hi, 2),
            "dit_low_noise_s": round(dit_lo, 2),
            "vae_decode_s": round(probe.vae.seconds, 3),
            "other_s": round(
                max(0.0, gen_s - probe.text.seconds - dit_hi - dit_lo - probe.vae.seconds),
                2,
            ),
            "dit_calls": probe.dit.calls + probe.dit2.calls,
            "dit_high_noise_calls": probe.dit.calls,
            "dit_low_noise_calls": probe.dit2.calls,
            "text_encode_calls": probe.text.calls,
            "vae_decode_calls": probe.vae.calls,
        }

    if cpu_run:
        import resource
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        peak_gib = (rss_after - rss_before) / 1024**3
    else:
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = _save_video([f.pixels for f in frames], out, fps=NATIVE_FPS)

    summary = {
        "model": engine.model_name,
        "repo": config.repo_id,
        "device": device.name,
        "frames": len(frames),
        "resolution": f"{args.width}x{args.height}",
        "steps": args.steps,
        "load_seconds": round(load_s, 1),
        "generate_seconds": round(gen_s, 1),
        "frames_per_second": round(len(frames) / gen_s, 3) if gen_s else 0.0,
        "seconds_per_step": round(gen_s / args.steps, 2) if args.steps else 0.0,
        "peak_hbm_gib": round(peak_gib, 1),
        "output": str(saved),
    }
    print("[repercep] RESULT " + json.dumps(summary), flush=True)

    if prof_payload is not None:
        print("[repercep] PROFILE " + json.dumps(prof_payload), flush=True)
        dit_loop = float(prof_payload["dit_loop_s"])
        dit_share = (dit_loop / gen_s * 100) if gen_s else 0.0
        print(
            f"  total {gen_s:7.1f}s | "
            f"text {prof_payload['text_encode_s']:6.2f}s | "
            f"DiT {dit_loop:7.1f}s ({dit_share:.0f}%, "
            f"{prof_payload['dit_calls']} calls — "
            f"hi {prof_payload['dit_high_noise_s']:.1f}s / "
            f"{prof_payload['dit_high_noise_calls']} + "
            f"lo {prof_payload['dit_low_noise_s']:.1f}s / "
            f"{prof_payload['dit_low_noise_calls']}) | "
            f"VAE {prof_payload['vae_decode_s']:6.2f}s | "
            f"other {prof_payload['other_s']:6.1f}s",
            flush=True,
        )

    if args.profile_second_pass:
        # Separate warmup-separated steady-state profile. Doubles the script's
        # wall time; only useful when the primary run includes ROCm kernel
        # autotuning cost (cold cache) that we want excluded from the
        # breakdown. The inline --profile above is sufficient otherwise.
        print(
            f"[repercep] second-pass profiling ({args.frames}f / {args.steps} steps) ...",
            flush=True,
        )
        prof = profile_wan(engine, request, warmup=1)
        print("[repercep] PROFILE2 " + json.dumps(prof.model_dump()), flush=True)
        print(
            f"  total {prof.total_s:7.1f}s | "
            f"text {prof.text_encode_s:6.2f}s | "
            f"DiT {prof.dit_loop_s:7.1f}s ({prof.dit_share * 100:.0f}%, "
            f"{prof.dit_calls} calls — "
            f"hi {prof.dit_high_noise_s:.1f}s / {prof.dit_high_noise_calls} + "
            f"lo {prof.dit_low_noise_s:.1f}s / {prof.dit_low_noise_calls}) | "
            f"VAE {prof.vae_decode_s:6.2f}s | other {prof.other_s:6.1f}s",
            flush=True,
        )
    return 0


def _save_video(frame_tensors: list, path: Path, fps: int = 16) -> Path:
    """Write frames to mp4 via imageio; fall back to PNG frames via PIL."""
    import numpy as np

    stacked = np.stack([t.numpy() for t in frame_tensors])  # (T, H, W, 3) uint8
    try:
        import imageio.v3 as iio

        iio.imwrite(path, stacked, fps=fps, codec="libx264")
        return path
    except Exception as exc:
        print(f"[repercep] mp4 encode unavailable ({exc}); writing PNG frames", flush=True)
        from PIL import Image

        frame_dir = path.with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(stacked):
            Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")
        return frame_dir


if __name__ == "__main__":
    raise SystemExit(main())
