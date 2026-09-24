#!/usr/bin/env python3
"""Measure CEM-candidate batching for V-JEPA 2-AC energy-MPC planning.

Baseline (what the engine ships today, ``VJepa2ACEngine._plan_sequence``): the
``S`` candidate action sequences are rolled out SEQUENTIALLY, one predictor
forward per (candidate, timestep) at batch=1 -> ``S * H * iters`` forwards.

Optimized: roll out ALL ``S`` candidates in ONE batched predictor forward per
timestep -> ``H * iters`` forwards (batch=``S``). Identical CEM math and refit;
only the rollout is batched across candidates. This is the action-loop serving
mechanism the revised strategy names (CEM-candidate batching). We report
wall-time for both and the planned-trajectory energy of each so the speedup is
not bought with a worse plan.

    python scripts/bench_cem_batched.py --horizons 4,8

Finding (1x H100, 64 samples x 3 iters): ~1.6x, NOT the naive ``S``x — the
per-candidate forward over the ~2048-token context is already compute-bound, so
batching buys GPU efficiency, not launch-overhead elimination. See
``docs/BENCH_2026_06_RUNPOD.md``.
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
import torch.nn.functional as F  # noqa: E402, N812

from repercep.backend.registry import select_backend  # noqa: E402
from repercep.models.vjepa2_ac import VJepa2ACConfig, VJepa2ACEngine  # noqa: E402
from repercep.runtime.types import Action, ConditioningInput, RolloutParams  # noqa: E402

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def batched_plan_sequence(engine, state, goal, horizon, seed=0):
    """CEM with the rollout batched across candidates. Returns (mean_seq, elapsed_s).

    This intentionally reaches into the engine's predictor adapter so the
    measurement uses the *same weights and dtype* as the shipped sequential path
    — the only delta is the batch dimension over candidates.
    """
    cfg = engine._config
    tokens_per_frame = engine._tokens_per_frame
    adim = cfg.action_dim
    n_samples, n_elites, iters = cfg.plan_samples, cfg.plan_elites, cfg.plan_iters
    keep = cfg.context_frames * tokens_per_frame
    adapter = engine._predictor
    predictor = adapter._predictor
    cdtype = adapter._cdtype or state.context.dtype

    base_ctx = state.context.to(cdtype)
    goal_c = goal.to(cdtype)
    dev = base_ctx.device
    g = torch.Generator(device=dev).manual_seed(seed)
    mean = torch.zeros(horizon, adim, dtype=cdtype, device=dev)
    std = torch.ones_like(mean)

    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        noise = torch.randn(n_samples, horizon, adim, generator=g, dtype=cdtype, device=dev)
        seqs = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        ctx = base_ctx.unsqueeze(0).expand(n_samples, -1, -1).contiguous()
        with torch.inference_mode():
            for t in range(horizon):
                n_frames = ctx.shape[1] // tokens_per_frame
                actions = ctx.new_zeros(n_samples, n_frames, adim)
                actions[:, -1] = seqs[:, t]
                states = torch.zeros_like(actions)
                out = predictor(ctx, actions, states)
                tokens = out if isinstance(out, torch.Tensor) else out.last_hidden_state
                block = F.layer_norm(tokens[:, -tokens_per_frame:], (tokens.shape[-1],))
                ctx = torch.cat([ctx, block], dim=1)[:, -keep:]
            terminal = ctx[:, -tokens_per_frame:]
            energies = torch.linalg.vector_norm(
                (terminal - goal_c.unsqueeze(0)).reshape(n_samples, -1), dim=1
            )
        elite_idx = torch.topk(energies, n_elites, largest=False).indices
        elite = seqs[elite_idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(1e-6)
    _sync()
    return mean, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    ap.add_argument("--horizons", default="4,8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    cfg = VJepa2ACConfig(action_dim=7, context_frames=8, dtype=_DTYPES[args.dtype])
    engine = VJepa2ACEngine(backend, cfg)
    print("[opt] loading engine (weights cached) ...", flush=True)
    t = time.perf_counter()
    engine.load()
    _sync()
    print(
        f"[opt] loaded in {time.perf_counter() - t:.1f}s  P={engine._tokens_per_frame}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    state = engine.reset(ConditioningInput(), RolloutParams(horizon=4))

    out: dict = {
        "model": engine.model_name,
        "device": backend.devices()[0].name,
        "dtype": args.dtype,
        "cem": {"samples": cfg.plan_samples, "elites": cfg.plan_elites, "iters": cfg.plan_iters},
        "results": {},
    }
    for horizon in [int(x) for x in args.horizons.split(",")]:
        gs, _ = engine.step(state, Action(values=torch.randn(7).tolist()))
        goal = gs.context[-engine._tokens_per_frame :]

        _sync()
        t = time.perf_counter()
        seq_mean = engine._plan_sequence(state, goal, horizon)
        _sync()
        seq_s = time.perf_counter() - t
        e_seq = float(engine._rollout_energy(state, seq_mean, goal))

        bat_mean, bat_s = batched_plan_sequence(engine, state, goal, horizon)
        e_bat = float(engine._rollout_energy(state, bat_mean.to(state.context.dtype), goal))

        speedup = seq_s / bat_s if bat_s else 0.0
        out["results"][str(horizon)] = {
            "sequential_s": round(seq_s, 3),
            "batched_s": round(bat_s, 3),
            "speedup": round(speedup, 1),
            "energy_sequential": round(e_seq, 3),
            "energy_batched": round(e_bat, 3),
            "forwards_sequential": cfg.plan_samples * cfg.plan_iters * horizon,
            "forwards_batched": cfg.plan_iters * horizon,
        }
        print(
            f"[opt] H={horizon}: seq {seq_s:.2f}s -> batched {bat_s:.2f}s = {speedup:.1f}x "
            f"| energy {e_seq:.2f} vs {e_bat:.2f}",
            flush=True,
        )

    out["peak_hbm_gib"] = (
        round(torch.cuda.max_memory_allocated() / 1024**3, 1) if torch.cuda.is_available() else 0.0
    )
    print("[opt] RESULT " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
