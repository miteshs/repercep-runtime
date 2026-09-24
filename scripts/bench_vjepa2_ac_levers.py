#!/usr/bin/env python3
"""Measure the 2026-07 latency levers on the REAL V-JEPA 2-AC engine.

Extends ``scripts/bench_vjepa2_ac.py`` (the sequential-fp32 June baseline) with
the three levers that landed in ``models/vjepa2_ac.py``:

- **batched candidates** (``plan_batched``): one predictor forward per rollout
  timestep instead of per candidate;
- **bf16 predictor compute** (``predictor_compute_dtype="bfloat16"`` + the SDPA
  dtype harmonizer) — this run IS the "VERIFY ON GPU" for that path, so it
  first checks energy parity against fp32 on identical action sequences;
- **warm start** (``plan_warm_start``): reported as achieved-energy at a fixed
  iteration budget (shift-reuse changes convergence, not per-iter cost).

Emits one verbatim ``RESULT`` JSON line (repo provenance convention).

    python scripts/bench_vjepa2_ac_levers.py                # H100/ROCm box
    python scripts/bench_vjepa2_ac_levers.py --skip-sequential  # reuse June 55.6s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _setup() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))


_setup()

import torch  # noqa: E402

from repercep.backend.registry import select_backend  # noqa: E402
from repercep.models.vjepa2_ac import VJepa2ACConfig, VJepa2ACEngine  # noqa: E402
from repercep.runtime.types import Action, ConditioningInput, RolloutParams  # noqa: E402


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_plan(engine: VJepa2ACEngine, state: Any, goal: torch.Tensor, horizon: int) -> float:
    _sync()
    t = time.perf_counter()
    engine.plan(state, goal, horizon=horizon)
    _sync()
    return time.perf_counter() - t


def _step_ms(engine: VJepa2ACEngine, state: Any, iters: int) -> tuple[Any, float]:
    for _ in range(3):
        state, _ = engine.step(state, Action(values=torch.randn(7).tolist()))
    _sync()
    t = time.perf_counter()
    for _ in range(iters):
        state, _ = engine.step(state, Action(values=torch.randn(7).tolist()))
    _sync()
    return state, (time.perf_counter() - t) / iters * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--step-iters", type=int, default=20)
    ap.add_argument(
        "--skip-sequential",
        action="store_true",
        help="skip the ~55s sequential-fp32 baseline (June number stands)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    dev = backend.devices()[0]
    print(f"[levers] backend={backend.name} device={dev.name}", flush=True)

    engines: dict[str, VJepa2ACEngine] = {}
    load_s: dict[str, float] = {}
    for name in ("float32", "bfloat16"):
        cfg = VJepa2ACConfig(predictor_compute_dtype=name)
        engines[name] = VJepa2ACEngine(backend, cfg)
        t = time.perf_counter()
        engines[name].load()
        _sync()
        load_s[name] = round(time.perf_counter() - t, 1)
        print(f"[levers] load {name}: {load_s[name]}s", flush=True)

    fp32, bf16 = engines["float32"], engines["bfloat16"]
    state32 = fp32.reset(ConditioningInput(), RolloutParams(horizon=args.horizon))
    state16 = bf16.reset(ConditioningInput(), RolloutParams(horizon=args.horizon))

    # --- 1. bf16 parity: identical action sequences, energy relative diff ---
    torch.manual_seed(args.seed)
    p = fp32._tokens_per_frame
    goal32 = state32.context[-p:]
    goal16 = state16.context[-p:]
    seqs = torch.randn(8, args.horizon, 7, device=state32.context.device).to(state32.context.dtype)
    e32 = fp32._candidate_energies(state32, seqs, goal32)
    e16 = bf16._candidate_energies(state16, seqs.to(state16.context.dtype), goal16)
    rel = ((e32.float() - e16.float()).abs() / e32.float().clamp_min(1e-6)).max().item()
    print(f"[levers] parity: max rel energy diff fp32 vs bf16 = {rel:.4f}", flush=True)

    # --- 2. step latency ---
    state32, step32_ms = _step_ms(fp32, state32, args.step_iters)
    state16, step16_ms = _step_ms(bf16, state16, args.step_iters)
    print(f"[levers] step fp32 {step32_ms:.1f} ms | bf16 {step16_ms:.1f} ms", flush=True)

    # --- 3. plan-latency ladder (H=horizon, default 64x3 CEM budget) ---
    goal32 = state32.context[-p:]
    goal16 = state16.context[-p:]
    ladder: dict[str, float] = {}

    if not args.skip_sequential:
        fp32._config.plan_batched = False
        fp32._config.plan_warm_start = False
        fp32._plan_mean.clear()
        ladder["sequential_fp32_s"] = round(_timed_plan(fp32, state32, goal32, args.horizon), 2)
        print(f"[levers] plan seq fp32: {ladder['sequential_fp32_s']}s", flush=True)

    fp32._config.plan_batched = True
    fp32._config.plan_warm_start = False
    fp32._plan_mean.clear()
    ladder["batched_fp32_s"] = round(_timed_plan(fp32, state32, goal32, args.horizon), 2)
    print(f"[levers] plan batched fp32: {ladder['batched_fp32_s']}s", flush=True)

    bf16._config.plan_batched = True
    bf16._config.plan_warm_start = False
    bf16._plan_mean.clear()
    ladder["batched_bf16_s"] = round(_timed_plan(bf16, state16, goal16, args.horizon), 2)
    print(f"[levers] plan batched bf16: {ladder['batched_bf16_s']}s", flush=True)

    # --- 4. warm start: achieved terminal energy at a 1-iter budget ---
    torch.manual_seed(args.seed)
    bf16._config.plan_warm_start = True
    bf16._plan_mean.clear()
    full = bf16._plan_sequence(state16, goal16, args.horizon)  # seeds the warm cache
    e_full = bf16._candidate_energies(state16, full.unsqueeze(0), goal16).item()
    bf16._config.plan_iters, saved_iters = 1, bf16._config.plan_iters
    warm1 = bf16._plan_sequence(state16, goal16, args.horizon)
    e_warm1 = bf16._candidate_energies(state16, warm1.unsqueeze(0), goal16).item()
    bf16._plan_mean.clear()
    cold1 = bf16._plan_sequence(state16, goal16, args.horizon)
    e_cold1 = bf16._candidate_energies(state16, cold1.unsqueeze(0), goal16).item()
    bf16._config.plan_iters = saved_iters
    print(
        f"[levers] energy: full(3it)={e_full:.4f} warm(1it)={e_warm1:.4f} cold(1it)={e_cold1:.4f}",
        flush=True,
    )

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    result = {
        "bench": "vjepa2_ac_levers",
        "device": dev.name,
        "horizon": args.horizon,
        "cem_budget": {"samples": 64, "iters": 3},
        "load_seconds": load_s,
        "parity_max_rel_energy_diff_bf16": round(rel, 5),
        "step_ms": {"fp32": round(step32_ms, 1), "bf16": round(step16_ms, 1)},
        "plan_ladder": ladder,
        "warm_start_energy": {
            "full_3iter": round(e_full, 4),
            "warm_1iter": round(e_warm1, 4),
            "cold_1iter": round(e_cold1, 4),
        },
        "peak_hbm_gib": round(peak, 1),
    }
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
