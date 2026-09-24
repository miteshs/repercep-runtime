#!/usr/bin/env python3
"""Per-stage profile of a Cosmos-Predict-7B generation on MI300X.

    python scripts/profile_cosmos.py --frames 49 --steps 12             # baseline
    python scripts/profile_cosmos.py --frames 49 --steps 12 --compile   # torch.compile
    python scripts/profile_cosmos.py --frames 49 --steps 12 --compare   # both + speedup

The profile splits wall time into text-encode / DiT denoising loop / VAE decode
/ other, so optimization targets the measured bottleneck (docs/OPTIMIZATION.md).
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-stage Cosmos profile")
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="torch.compile the DiT")
    parser.add_argument("--compare", action="store_true", help="baseline vs compiled")
    parser.add_argument(
        "--native-loop",
        action="store_true",
        help="Repercep-native denoising loop (CFG batching)",
    )
    parser.add_argument(
        "--cache-skip-every",
        type=int,
        default=0,
        help="skip the DiT forward N-1 of every N steps (native loop only)",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("none", "fixed", "adaptive"),
        default="none",
        help=(
            "caching strategy: 'fixed' uses --cache-skip-every; "
            "'adaptive' uses input-similarity gating"
        ),
    )
    parser.add_argument(
        "--cache-adaptive-threshold",
        type=float,
        default=0.3,
        help="adaptive cache: accumulated rel-L1 threshold for triggering a full forward",
    )
    parser.add_argument(
        "--cache-force-full-every",
        type=int,
        default=16,
        help="adaptive cache: force a full forward at least every N steps (0=disabled)",
    )
    parser.add_argument("--prompt", default="A drone shot flying over a coastal highway at sunset.")
    args = parser.parse_args()

    from repercep.backend.registry import select_backend
    from repercep.bench.profile import profile_cosmos
    from repercep.models.cosmos import CosmosConfig, CosmosEngine
    from repercep.runtime.types import GenerationParams, GenerationRequest

    backend = select_backend()
    request = GenerationRequest(
        prompt=args.prompt,
        params=GenerationParams(
            num_frames=args.frames,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            seed=0,
        ),
    )
    print(
        f"[repercep] profiling Cosmos: {args.frames}f / {args.steps} steps, warmup={args.warmup}",
        flush=True,
    )

    if args.compare:
        base = profile_cosmos(
            CosmosEngine(
                backend,
                CosmosConfig(
                    use_native_loop=args.native_loop,
                    cache_skip_every=args.cache_skip_every,
                    cache_mode=args.cache_mode,
                    cache_adaptive_threshold=args.cache_adaptive_threshold,
                    cache_force_full_every=args.cache_force_full_every,
                ),
            ),
            request,
            warmup=args.warmup,
        )
        _report("baseline", base)
        comp = profile_cosmos(
            CosmosEngine(
                backend,
                CosmosConfig(
                    compile_transformer=True,
                    use_native_loop=args.native_loop,
                    cache_skip_every=args.cache_skip_every,
                    cache_mode=args.cache_mode,
                    cache_adaptive_threshold=args.cache_adaptive_threshold,
                    cache_force_full_every=args.cache_force_full_every,
                ),
            ),
            request,
            warmup=args.warmup,
        )
        _report("compiled", comp)
        print(
            f"[repercep] SPEEDUP  DiT loop {base.dit_loop_s / comp.dit_loop_s:.2f}x   "
            f"end-to-end {base.total_s / comp.total_s:.2f}x"
        )
    else:
        engine = CosmosEngine(
            backend,
            CosmosConfig(
                compile_transformer=args.compile,
                use_native_loop=args.native_loop,
                cache_skip_every=args.cache_skip_every,
                cache_mode=args.cache_mode,
                cache_adaptive_threshold=args.cache_adaptive_threshold,
                cache_force_full_every=args.cache_force_full_every,
            ),
        )
        label_parts = []
        if args.native_loop:
            label_parts.append("native")
        label_parts.append("compiled" if args.compile else "baseline")
        _report("+".join(label_parts), profile_cosmos(engine, request, warmup=args.warmup))
    return 0


def _report(label, profile):
    print(f"[repercep] PROFILE {label} " + json.dumps(profile.model_dump()), flush=True)
    print(
        f"  [{label}] total {profile.total_s:7.1f}s | "
        f"text {profile.text_encode_s:6.2f}s | "
        f"DiT {profile.dit_loop_s:7.1f}s ({profile.dit_share * 100:.0f}%, "
        f"{profile.dit_calls} calls) | "
        f"VAE {profile.vae_decode_s:6.2f}s | other {profile.other_s:6.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
