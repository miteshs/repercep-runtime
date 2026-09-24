#!/usr/bin/env python3
"""Latency/throughput benchmark for the REAL V-JEPA 2-AC engine.

``scripts/run_vjepa2_ac.py`` is a *wiring* demo — it prints context shapes and a
single energy delta, not timings. This script measures the numbers that matter
for the interactive control regime (ADR-0008): encoder ``reset`` latency, the
per-step latent rollout latency, and the cost of a CEM/MPC ``plan`` call
(``samples * iters * horizon`` predictor forwards). Emits one JSON ``RESULT``
line, mirroring ``scripts/run_cosmos.py``.

    python scripts/bench_vjepa2_ac.py                 # real weights, GPU
    python scripts/bench_vjepa2_ac.py --dtype fp32    # ROCm: fp32 encoder (MIOpen)

The CEM ``plan`` here is the *sequential* baseline the engine ships
(``_plan_sequence`` rolls candidates out one at a time at batch=1). See
``scripts/bench_cem_batched.py`` for the candidate-batched variant and the
measured speedup.
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
from repercep.models.vjepa2_ac import VJepa2ACConfig, VJepa2ACEngine  # noqa: E402
from repercep.runtime.types import Action, ConditioningInput, RolloutParams  # noqa: E402

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    ap.add_argument("--step-iters", type=int, default=20, help="timed rollout steps")
    ap.add_argument("--plan-horizons", default="4,8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    dev = backend.devices()[0]
    cfg = VJepa2ACConfig(action_dim=7, context_frames=8, dtype=_DTYPES[args.dtype])
    engine = VJepa2ACEngine(backend, cfg)

    print(f"[bench] backend={backend.name} device={dev.name} dtype={args.dtype}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t = time.perf_counter()
    engine.load()
    _sync()
    load_s = time.perf_counter() - t
    print(f"[bench] load {load_s:.1f}s  tokens_per_frame={engine._tokens_per_frame}", flush=True)

    t = time.perf_counter()
    state = engine.reset(ConditioningInput(), RolloutParams(horizon=4))
    _sync()
    reset_s = time.perf_counter() - t
    print(f"[bench] reset {reset_s:.3f}s context={tuple(state.context.shape)}", flush=True)

    # warmup then time the rollout step
    for _ in range(3):
        state, _ = engine.step(state, Action(values=torch.randn(7).tolist()))
    _sync()
    t = time.perf_counter()
    for _ in range(args.step_iters):
        state, _ = engine.step(state, Action(values=torch.randn(7).tolist()))
    _sync()
    step_s = (time.perf_counter() - t) / args.step_iters
    print(f"[bench] step {step_s * 1000:.1f} ms/step  ({1 / step_s:.1f} steps/s)", flush=True)

    plans: dict[str, dict[str, float]] = {}
    for horizon in [int(x) for x in args.plan_horizons.split(",")]:
        gs, _ = engine.step(state, Action(values=torch.randn(7).tolist()))
        goal = gs.context[-engine._tokens_per_frame :]
        _sync()
        t = time.perf_counter()
        engine.plan(state, goal, horizon=horizon)
        _sync()
        plan_s = time.perf_counter() - t
        fwds = cfg.plan_samples * cfg.plan_iters * horizon
        plans[str(horizon)] = {
            "plan_seconds": round(plan_s, 3),
            "predictor_forwards": fwds,
            "ms_per_forward": round(plan_s / fwds * 1000, 2),
        }
        print(
            f"[bench] plan H={horizon}: "
            f"{plan_s:.2f}s ({fwds} fwds, {plan_s / fwds * 1000:.1f} ms/fwd)",
            flush=True,
        )

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    result = {
        "model": engine.model_name,
        "device": dev.name,
        "dtype": args.dtype,
        "load_seconds": round(load_s, 1),
        "reset_seconds": round(reset_s, 3),
        "step_ms": round(step_s * 1000, 1),
        "steps_per_sec": round(1 / step_s, 1),
        "cem_config": {"samples": cfg.plan_samples, "iters": cfg.plan_iters},
        "plans": plans,
        "peak_hbm_gib": round(peak, 1),
        "context_shape": list(state.context.shape),
    }
    print("[bench] RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
