#!/usr/bin/env python3
"""Measure candidate batching for token-VLA action planning (Plan B, Phase 2).

Baseline: the planner decodes ``N`` candidate action-token sequences with a
per-candidate loop -> ``N`` autoregressive decodes at batch=1.

Optimized: decode all ``N`` candidates in ONE batched forward that shares the
prompt+observation prefix KV -> 1 decode at batch=``N``. Identical candidate
set and scoring; only the decode is batched. This is the token-VLA analogue of
the V-JEPA-AC CEM-candidate-batching lever (``scripts/bench_cem_batched.py``),
and the leaderboard row the revised strategy names — one neither vLLM nor NIM
optimizes, on any silicon.

    python scripts/bench_vla_levers.py --candidates 8,16 --repeats 20

STATUS: Phase-2 artifact, pending the Phase-1 GPU port (``docs/VLA_PORT_PLAN.md``).
``VLAEngine.load()`` raises the port recipe until ``_VLAPipeline`` is real, so
this script runs end-to-end only on a box where Phase 1 has landed. It reports
wall-time for both paths so the batched speedup is measured, never fabricated
(``docs/METHODOLOGY.md`` §3) — there is deliberately no synthetic-timing path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _setup() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))


_setup()

import torch  # noqa: E402

from repercep.backend.registry import select_backend  # noqa: E402
from repercep.models.vla import VLAConfig, VLAEngine  # noqa: E402
from repercep.runtime.types import Action, ConditioningInput, RolloutParams  # noqa: E402

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_plan(engine, state, goal, horizon, *, batched, repeats):
    """Median wall-time of one plan() under the given decode path.

    Flips only ``plan_batched`` on the shared, already-loaded engine, so both
    paths use the same weights, pipeline, and candidate count — the only delta
    is the batch dimension over candidates.
    """
    engine._config.plan_batched = batched
    engine.plan(state, goal, horizon)  # warm up (compile / allocator)
    _sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.plan(state, goal, horizon)
        _sync()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    ap.add_argument("--repo", default=None, help="Model bundle (default: VLAConfig.repo).")
    ap.add_argument("--candidates", default="8,16", help="Comma-separated candidate counts.")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    cfg_kwargs = {"dtype": _DTYPES[args.dtype]}
    if args.repo is not None:
        cfg_kwargs["repo"] = args.repo
    engine = VLAEngine(backend, VLAConfig(**cfg_kwargs))

    print("[vla] loading engine ...", flush=True)
    try:
        engine.load()
    except NotImplementedError as exc:
        # The honest gate: no Phase-1 pipeline on this box -> nothing to measure.
        print(f"[vla] Phase-1 pipeline not available; cannot bench: {exc}", flush=True)
        return 2
    _sync()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    state = engine.reset(ConditioningInput(), RolloutParams())
    # A neutral goal of the context width; the real scorer defines its meaning.
    goal = state.context.new_zeros(state.context.shape[-1])
    # Prime one step so plan() runs against an advanced context, not just reset.
    state, _ = engine.step(state, Action(values=[0.0] * engine._config.action_dim))

    out: dict = {
        "model": engine.model_name,
        "device": backend.devices()[0].name,
        "dtype": args.dtype,
        "horizon": args.horizon,
        "repeats": args.repeats,
        "results": {},
    }
    for n in [int(x) for x in args.candidates.split(",")]:
        engine._config.plan_candidates = n
        loop_s = _time_plan(engine, state, goal, args.horizon, batched=False, repeats=args.repeats)
        bat_s = _time_plan(engine, state, goal, args.horizon, batched=True, repeats=args.repeats)
        speedup = loop_s / bat_s if bat_s else 0.0
        out["results"][str(n)] = {
            "per_candidate_s": round(loop_s, 4),
            "batched_s": round(bat_s, 4),
            "speedup": round(speedup, 2),
            "decodes_per_candidate": n,
            "decodes_batched": 1,
        }
        print(
            f"[vla] N={n}: loop {loop_s * 1e3:.1f}ms -> batched {bat_s * 1e3:.1f}ms "
            f"= {speedup:.2f}x",
            flush=True,
        )

    out["peak_hbm_gib"] = (
        round(torch.cuda.max_memory_allocated() / 1024**3, 1) if torch.cuda.is_available() else 0.0
    )
    print("[vla] RESULT " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
