#!/usr/bin/env python3
"""Verify Wan-2.2 H100 smoke timing is reproducible.

Runs ``scripts/run_wan.py --frames 17 --steps 8 --vae-tiling`` N times with
distinct seeds and reports mean, min, max, std for ``generate_seconds`` and
``peak_hbm_gib``. Sibling of ``scripts/verify_timing.py`` (which is Cosmos-
specific). The 81 f / 40 step shape is too long for routine variance runs
(~26 min x N = hours); for smoke variance, 17 f / 8 step at ~38 s warm is
right-sized.

Usage::

    REPERCEP_FP8_ATTENTION=fa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        HF_HOME=/workspace/hf-cache \\
        .venv/bin/python scripts/verify_wan_timing.py --N 5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROMPTS_AND_SEEDS: list[tuple[str, int]] = [
    (
        "A sleek autonomous delivery robot rolls along a sunlit city sidewalk "
        "past glass storefronts, smooth forward motion, photorealistic, high detail.",
        0,
    ),
    (
        "A small fishing boat cuts through choppy ocean waves at sunset, "
        "spray catching golden light, cinematic camera tracking from the side.",
        42,
    ),
    (
        "An overhead drone shot of dense rainforest canopy, slowly drifting "
        "forward as morning fog lifts between the treetops, photorealistic.",
        100,
    ),
    (
        "A wide-angle locked-off shot of a busy night market street in Tokyo, "
        "neon signs reflecting on wet pavement, people walking past in slow motion.",
        7,
    ),
    (
        "A first-person POV walking up a steep mountain trail, loose gravel "
        "underfoot, jagged peaks ahead, crisp early-morning light.",
        13,
    ),
]


def _run(cmd: list[str]) -> tuple[float, dict[str, Any] | None]:
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall = time.monotonic() - t0
    result: dict[str, Any] | None = None
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        if line.startswith("[repercep] RESULT "):
            with contextlib.suppress(json.JSONDecodeError):
                result = json.loads(line[len("[repercep] RESULT "):])
    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
    return wall, result


def _build_cmd(prompt: str, seed: int, out_path: str, *, frames: int, steps: int) -> list[str]:
    return [
        ".venv/bin/python",
        "scripts/run_wan.py",
        "--prompt", prompt,
        "--seed", str(seed),
        "--frames", str(frames),
        "--steps", str(steps),
        "--vae-tiling",
        "--out", out_path,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=5,
                        help="number of (prompt, seed) pairs to run (1..5)")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    n = max(1, min(args.N, len(PROMPTS_AND_SEEDS)))
    pairs = PROMPTS_AND_SEEDS[:n]

    runs: list[dict[str, Any]] = []
    for i, (prompt, seed) in enumerate(pairs, 1):
        out = f"benchmark-results/wan_variance_{i:02d}_seed{seed}.mp4"
        print(f"[verify_wan] run {i}/{n} seed={seed} prompt={prompt[:48]!r} ...", flush=True)
        wall, result = _run(_build_cmd(prompt, seed, out, frames=args.frames, steps=args.steps))
        if result is None:
            print(f"  no RESULT line; skipping aggregation for run {i}", file=sys.stderr)
            continue
        result["_wall_subprocess_s"] = round(wall, 2)
        result["_seed"] = seed
        runs.append(result)
        print(f"  gen={result['generate_seconds']:.1f}s  peak={result['peak_hbm_gib']:.1f} GiB  "
              f"(wall {wall:.1f}s)", flush=True)

    if not runs:
        print("[verify_wan] no successful runs; aborting", file=sys.stderr)
        return 1

    gens = [float(r["generate_seconds"]) for r in runs]
    peaks = [float(r["peak_hbm_gib"]) for r in runs]
    summary = {
        "config": f"{args.frames}f/{args.steps}/vae_tiling=True",
        "n_runs": len(runs),
        "generate_seconds": {
            "mean": round(statistics.fmean(gens), 2),
            "stdev": round(statistics.stdev(gens), 3) if len(gens) > 1 else 0.0,
            "min": round(min(gens), 2),
            "max": round(max(gens), 2),
            "pct_spread": round(100 * (max(gens) - min(gens)) / statistics.fmean(gens), 2),
        },
        "peak_hbm_gib": {
            "mean": round(statistics.fmean(peaks), 2),
            "stdev": round(statistics.stdev(peaks), 3) if len(peaks) > 1 else 0.0,
            "min": round(min(peaks), 2),
            "max": round(max(peaks), 2),
        },
        "runs": runs,
    }
    print()
    print("=== SUMMARY ===")
    print(f"  generate_seconds  mean={summary['generate_seconds']['mean']}s  "
          f"stdev={summary['generate_seconds']['stdev']}s  "
          f"range=[{summary['generate_seconds']['min']}, {summary['generate_seconds']['max']}]  "
          f"spread={summary['generate_seconds']['pct_spread']}%")
    print(f"  peak_hbm_gib      mean={summary['peak_hbm_gib']['mean']} GiB  "
          f"stdev={summary['peak_hbm_gib']['stdev']} GiB")

    ts = int(time.time())
    out_path = Path(f"benchmark-results/verify_wan_timing_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[verify_wan] sidecar -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
