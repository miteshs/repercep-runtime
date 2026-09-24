#!/usr/bin/env python3
"""Verify the headline Cosmos timing is reproducible.

Runs the 121 f / 36 step adaptive-cache config across:
  * N repeats of (default prompt, seed 0) — measures run-to-run variance
  * 5 distinct (prompt, seed) pairs — measures workload variance

Reports mean, min, max, std for each axis + the per-run JSON RESULT line
the engine emits. Designed to be run **after** Phase-2 work has landed and
the GPU is quiet (no other Python processes); see docs/METHODOLOGY.md §4.

Usage:
    .venv/bin/python scripts/verify_timing.py --N 3
    .venv/bin/python scripts/verify_timing.py --N 3 --prompts 5
    .venv/bin/python scripts/verify_timing.py --quick     # 1 repeat, 1 prompt — smoke

The output goes to stdout and to ``benchmark-results/verify_timing_<ts>.json``.
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

# Five distinct prompts spanning different scene compositions, to exercise the
# T5 encode + DiT loop across varied conditioning. Seeds chosen for diversity
# (mod 7 prime-ish) so PRNG state never lines up across pairs.
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
    """Run a generate command, return (wall_seconds_from_script, RESULT dict)."""
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
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


def _build_cmd(prompt: str, seed: int, out_path: str, *, no_cache: bool = False) -> list[str]:
    cache_args = (
        ["--cache-mode", "none"]
        if no_cache
        else [
            "--cache-mode", "adaptive",
            "--cache-adaptive-threshold", "0.30",
            "--cache-force-full-every", "16",
        ]
    )
    return [
        ".venv/bin/python",
        "scripts/run_cosmos.py",
        "--prompt", prompt,
        "--seed", str(seed),
        "--frames", "121",
        "--steps", "36",
        "--native-loop",
        *cache_args,
        "--out", out_path,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=3,
                        help="repeats of (prompt[0], seed[0]) for run-to-run variance")
    parser.add_argument("--prompts", type=int, default=1,
                        help=f"distinct (prompt, seed) pairs (1..{len(PROMPTS_AND_SEEDS)})")
    parser.add_argument("--no-cache-refs", action="store_true",
                        help="also run no-cache references at each of --prompts pairs "
                             "for FVD comparison (each ~470 s on MI300X — adds substantial time)")
    parser.add_argument("--skip-phase-a", action="store_true",
                        help="skip the N-repeats phase (determinism already verified by MD5)")
    parser.add_argument("--quick", action="store_true",
                        help="--N 1 --prompts 1 — single smoke run")
    args = parser.parse_args()

    if args.quick:
        args.N = 1
        args.prompts = 1
    if not (1 <= args.prompts <= len(PROMPTS_AND_SEEDS)):
        raise SystemExit(f"--prompts must be 1..{len(PROMPTS_AND_SEEDS)}")

    results: dict[str, Any] = {
        "config": "121f/36/adaptive thr=0.30 floor=16",
        "N_repeats_per_prompt": args.N,
        "n_prompts": args.prompts,
        "runs": [],
    }

    # 1. Repeats of the first (prompt, seed) — measures hardware/stack noise.
    base_prompt, base_seed = PROMPTS_AND_SEEDS[0]
    a_times: list[float] = []
    if not args.skip_phase_a:
        print(f"\n=== Phase A: {args.N} repeats of base prompt, seed {base_seed} ===")
        for i in range(args.N):
            out = f"benchmark-results/verify_baseA_run{i}.mp4"
            wall, r = _run(_build_cmd(base_prompt, base_seed, out))
            gen = (r or {}).get("generate_seconds")
            a_times.append(gen if gen is not None else wall)
            print(f"  run {i+1}/{args.N}: generate={gen}s  wall={wall:.1f}s  out={out}")
            results["runs"].append({
                "phase": "A_repeat", "i": i, "prompt_idx": 0,
                "wall": wall, "result": r,
            })
    else:
        print("\n=== Phase A skipped (--skip-phase-a) ===")

    # 2. Distinct (prompt, seed) pairs — measures workload variance.
    print(f"\n=== Phase B: {args.prompts} distinct (prompt, seed) pairs (adaptive) ===")
    b_times: list[float] = []
    for i, (prompt, seed) in enumerate(PROMPTS_AND_SEEDS[: args.prompts]):
        out = f"benchmark-results/verify_baseB_p{i}.mp4"
        wall, r = _run(_build_cmd(prompt, seed, out))
        gen = (r or {}).get("generate_seconds")
        b_times.append(gen if gen is not None else wall)
        print(f"  p{i} seed={seed}: generate={gen}s  wall={wall:.1f}s  out={out}")
        print(f"        prompt={prompt[:64]}...")
        results["runs"].append({
            "phase": "B_distinct", "i": i, "prompt_idx": i,
            "wall": wall, "result": r,
        })

    # 3. No-cache references at the same (prompt, seed) pairs — for FVD.
    c_times: list[float] = []
    if args.no_cache_refs:
        print(f"\n=== Phase C: no-cache refs at the {args.prompts} distinct pairs ===")
        for i, (prompt, seed) in enumerate(PROMPTS_AND_SEEDS[: args.prompts]):
            out = f"benchmark-results/verify_baseC_no_cache_p{i}.mp4"
            wall, r = _run(_build_cmd(prompt, seed, out, no_cache=True))
            gen = (r or {}).get("generate_seconds")
            c_times.append(gen if gen is not None else wall)
            print(f"  p{i} seed={seed}: generate={gen}s  wall={wall:.1f}s  out={out}")
            results["runs"].append({
                "phase": "C_no_cache_ref", "i": i, "prompt_idx": i,
                "wall": wall, "result": r,
            })

    # 3. Stats.
    def _stats(xs: list[float], label: str) -> None:
        if not xs:
            print(f"  {label}: no runs")
            return
        n = len(xs)
        mu = statistics.fmean(xs)
        sd = statistics.stdev(xs) if n > 1 else 0.0
        print(
            f"  {label}: N={n}  mean={mu:.2f}s  std={sd:.2f}s  "
            f"min={min(xs):.2f}s  max={max(xs):.2f}s"
        )

    print("\n=== Summary ===")
    _stats(a_times, "Phase A — same prompt+seed (adaptive)")
    _stats(b_times, "Phase B — distinct prompts (adaptive)")
    _stats(c_times, "Phase C — distinct prompts (no-cache)")
    combined_adaptive = a_times + b_times
    if combined_adaptive:
        _stats(combined_adaptive, "Combined adaptive (A+B)")

    results["summary"] = {
        "phase_A_generate_seconds": a_times,
        "phase_B_generate_seconds": b_times,
        "phase_C_no_cache_generate_seconds": c_times,
        "mean_adaptive_all": (
            statistics.fmean(combined_adaptive) if combined_adaptive else None
        ),
        "std_adaptive_all": (
            statistics.stdev(combined_adaptive) if len(combined_adaptive) > 1 else None
        ),
        "mean_no_cache_all": statistics.fmean(c_times) if c_times else None,
        "std_no_cache_all": statistics.stdev(c_times) if len(c_times) > 1 else None,
    }

    ts = int(time.time())
    summary_path = Path(f"benchmark-results/verify_timing_{ts}.json")
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
