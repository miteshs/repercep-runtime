#!/usr/bin/env python3
"""Control-Loop Serving Benchmark v0 — the leaderboard harness.

Drives ANY :class:`repercep.runtime.interactive.InteractiveWorldModel` through
the closed-loop protocol (``reset -> step* -> plan``) and measures the four
metrics `docs/CONTROL_LOOP_BENCH.md` defines as the control-regime serving
leaderboard nobody else publishes:

1. **Closed-loop step latency under state carryover** — warm ms per ``step``
   on one persistent session (the state carries; no re-encode per step).
2. **Planning-decisions/sec** — full ``plan()`` calls per second.
3. **Energy-evals/sec** — rollout-energy evaluations per second inside a plan
   (V-JEPA-class energy planners; ``null`` for policy-mode engines).
4. **Resident sessions per GPU** — measured single-session marginal HBM
   extrapolated against the device's total (weights counted once).

Emits one verbatim ``RESULT`` JSON line (repo provenance convention).

    python scripts/bench_control_loop.py --engine vjepa2-ac          # real weights, GPU
    python scripts/bench_control_loop.py --engine vjepa2-ac --fake   # CPU toy weights (CI)
    python scripts/bench_control_loop.py --engine lingbot-va \
        --obs-dir /workspace/lingbot-va/example/demo                # real weights, GPU
    python scripts/bench_control_loop.py --engine dreamzero \
        --obs-dir /workspace/dreamzero/example/droid                # real weights, GPU
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
from repercep.models.dreamzero import DreamZeroConfig, DreamZeroEngine  # noqa: E402
from repercep.models.lingbot_va import LingBotVAConfig, LingBotVAEngine  # noqa: E402
from repercep.models.vjepa2_ac import VJepa2ACConfig, VJepa2ACEngine  # noqa: E402
from repercep.runtime.types import (  # noqa: E402
    Action,
    ConditioningInput,
    ConditioningKind,
    RolloutParams,
)

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _ToyEncoder:
    """Deterministic (1, T, D) features — the CPU/CI stand-in for real weights."""

    def __init__(self, frames: int, dim: int) -> None:
        self._ctx = torch.arange(frames * dim, dtype=torch.float32).reshape(frames, dim) / dim

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self._ctx.unsqueeze(0)


class _ToyPredictor:
    """Linear toy dynamics: next frame = last context frame + action."""

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return context[-1] + action


def build_engine(args: argparse.Namespace) -> Any:
    """Construct the engine under test from CLI args."""
    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    if args.engine == "vjepa2-ac":
        cfg = VJepa2ACConfig(
            action_dim=args.action_dim,
            context_frames=8,
            dtype=_DTYPES[args.dtype],
            plan_samples=args.plan_samples,
            # Elites must not exceed samples (topk) — clamp for small CI budgets.
            plan_elites=min(8, args.plan_samples),
            plan_iters=args.plan_cem_iters,
        )
        if args.fake:
            return VJepa2ACEngine(
                backend,
                cfg,
                encoder=_ToyEncoder(frames=cfg.context_frames, dim=args.action_dim),
                predictor=_ToyPredictor(),
            )
        return VJepa2ACEngine(backend, cfg)
    # LingBot-VA / DreamZero: policy-regime engines (Phase-1 pipelines,
    # docs/LINGBOT_VA_PORT_PLAN.md §4 / docs/DREAMZERO_PORT_PLAN.md §4). Both
    # need a real prompt + seed-observation directory (--prompt/--obs-dir);
    # the fake path is unsupported (their rollout is a real chunked denoise
    # loop, not a toy-weights CPU stand-in).
    if args.fake:
        raise NotImplementedError(f"--fake is not supported for --engine {args.engine}")
    if args.engine == "dreamzero":
        return DreamZeroEngine(
            backend,
            DreamZeroConfig(
                prompt=args.prompt,
                cfg_scale=args.dz_cfg_scale,
                num_dit_steps=args.num_dit_steps,
                enable_dit_cache=args.dit_cache_dynamic,
                cfg_batched=args.cfg_batched,
                compile=args.compile,
                local_attn_size=args.attn_window,
            ),
        )
    return LingBotVAEngine(
        backend,
        LingBotVAConfig(
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            num_inference_steps=args.video_steps,
            action_num_inference_steps=args.action_steps,
            attn_mode=args.attn_mode,
            compile_transformer=args.compile,
        ),
    )


# Defaults mirrored from ``LingBotVAConfig`` — used to detect (and disclose)
# non-default lever flags in the RESULT line, per CONTROL_LOOP_BENCH.md §4's
# "disclose CFG scale, denoise step counts, attention path" submission rule.
_LINGBOT_VA_LEVER_DEFAULTS = {
    "guidance_scale": 5.0,
    "action_guidance_scale": 1.0,
    "video_steps": 5,
    "action_steps": 10,
    "attn_mode": "torch",
    "compile": False,
}

# Defaults mirrored from ``DreamZeroConfig`` (docs/DREAMZERO_PORT_PLAN.md §4)
# — note ``compile`` is a shared flag name with LingBot-VA's, disclosed under
# whichever engine is actually selected.
_DREAMZERO_LEVER_DEFAULTS = {
    "dz_cfg_scale": 5.0,
    "num_dit_steps": 16,
    "dit_cache_dynamic": False,
    "cfg_batched": False,
    "compile": False,
    "attn_window": None,
}


def lingbot_va_lever_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Non-default LingBot-VA lever flags, for RESULT-line disclosure."""
    if args.engine != "lingbot-va":
        return {}
    return {
        flag: getattr(args, flag)
        for flag, default in _LINGBOT_VA_LEVER_DEFAULTS.items()
        if getattr(args, flag) != default
    }


def dreamzero_lever_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Non-default DreamZero lever flags, for RESULT-line disclosure."""
    if args.engine != "dreamzero":
        return {}
    return {
        flag: getattr(args, flag)
        for flag, default in _DREAMZERO_LEVER_DEFAULTS.items()
        if getattr(args, flag) != default
    }


def _chunk_len(config: Any) -> int:
    """Executed-chunk row count for a chunked (policy-regime) engine's config.

    Dispatches on field presence rather than an engine-kind flag, matching
    the ``is_chunked``/``_wire_action_dim`` duck-typing already used here:
    LingBot-VA's chunk is ``frame_chunk_size x action_per_frame`` rows;
    DreamZero's is ``num_action_per_block`` rows directly (one action per
    block-frame-slot, not per-frame x per-frame-count).
    """
    if hasattr(config, "num_action_per_block"):
        return config.num_action_per_block
    return config.frame_chunk_size * config.action_per_frame


def bench_control_loop(
    engine: Any,
    *,
    warmup: int = 3,
    step_iters: int = 20,
    plan_calls: int = 1,
    horizon: int = 4,
    action_dim: int = 7,
    obs_dir: str | None = None,
    seed: int = 0,
    sessions: int = 1,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the four leaderboard metrics on one persistent session.

    ``action_dim``/random-action generation is V-JEPA-shaped (one control
    vector per ``step()``, advancing one latent frame). LingBot-VA's ``step()``
    is chunk-shaped instead (a full executed chunk — frame_chunk_size x
    action_per_frame rows of its wire action width — advances one denoise
    chunk); detected via ``_wire_action_dim`` (see ``repercep.models.lingbot_va``)
    so one harness drives both regimes without an engine-kind flag threaded
    through every call site.

    ``sessions`` (default 1, backward compatible): when >1, opens
    ``sessions - 1`` additional sessions on the SAME loaded engine after the
    metrics above (which stay computed on session 0 alone, unperturbed), then
    round-robins step() calls across all of them — a true N-measured resident
    curve alongside metric 4's single-session extrapolation, so the
    extrapolation's accuracy can be checked against reality (see
    ``result["concurrency"]``).

    ``extra``: caller-supplied fields merged into the result (e.g. disclosure
    of non-default LingBot-VA lever flags — CFG scale, denoise step counts,
    attention path — per CONTROL_LOOP_BENCH.md §4's submission rule).
    """
    torch.manual_seed(seed)
    on_gpu = torch.cuda.is_available()
    is_chunked = hasattr(engine, "_wire_action_dim")

    t = time.perf_counter()
    loader = getattr(engine, "load", None)
    if callable(loader) and not getattr(engine, "is_loaded", False):
        loader()
    _sync()
    load_s = time.perf_counter() - t
    baseline_gib = torch.cuda.memory_allocated() / 1024**3 if on_gpu else 0.0

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    conditioning = (
        ConditioningInput(kind=ConditioningKind.IMAGE, uri=obs_dir)
        if is_chunked
        else ConditioningInput()
    )
    state = engine.reset(conditioning, RolloutParams(horizon=horizon))
    _sync()
    reset_s = time.perf_counter() - t

    def rand_action() -> Action:
        if is_chunked:
            n = _chunk_len(engine._config)
            return Action(values=torch.randn(n * engine._wire_action_dim()).tolist())
        return Action(values=torch.randn(action_dim).tolist())

    # 1. closed-loop step latency under state carryover (warm).
    for _ in range(warmup):
        state, _ = engine.step(state, rand_action())
    _sync()
    t = time.perf_counter()
    for _ in range(step_iters):
        state, _ = engine.step(state, rand_action())
    _sync()
    step_s = (time.perf_counter() - t) / step_iters

    # 2./3. planning-decisions/sec and energy-evals/sec. The goal is a reached
    # state's terminal frame block (P patch-token rows; 1 on toy paths), so the
    # plan target is guaranteed feasible.
    tokens_per_frame = int(getattr(engine, "_tokens_per_frame", 1))
    goal_state, _ = engine.step(state, rand_action())
    goal = goal_state.context[-tokens_per_frame:]
    if is_chunked:
        # LingBot-VA's plan() reuses a chunk step() already parked
        # (session.pending_actions) instead of recomputing — correct for real
        # closed-loop use, but the step() above just parked one, which would
        # make the timed plan() below a free cache hit. Clear it so each timed
        # call does the real chunk denoise it's measuring.
        engine._sessions[state.session_id].pending_actions = None
    _sync()
    t = time.perf_counter()
    for _ in range(plan_calls):
        engine.plan(state, goal, horizon)
        if is_chunked:
            engine._sessions[state.session_id].pending_actions = None
    _sync()
    plan_s = (time.perf_counter() - t) / plan_calls

    cfg = getattr(engine, "_config", None)
    samples = getattr(cfg, "plan_samples", None)
    iters = getattr(cfg, "plan_iters", None)
    energy_evals = samples * iters if samples and iters else None

    # 4. resident sessions per GPU: marginal session HBM vs device total,
    # counting the (shared) weights once.
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if on_gpu else 0.0
    session_gib = max(peak_gib - baseline_gib, 0.0)
    if on_gpu and session_gib > 0:
        hbm_total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        resident_sessions = int((hbm_total_gib - baseline_gib) // session_gib)
    else:
        hbm_total_gib = 0.0
        resident_sessions = None

    # 5. (optional) N-live-session concurrency: a true measured curve, not the
    # metric-4 extrapolation from ONE session's marginal HBM. Opened AFTER the
    # metrics above so session 0's numbers stay exactly what they were before
    # this flag existed (sessions=1 is the unperturbed default path).
    concurrency: dict[str, Any] | None = None
    if sessions > 1:
        states = [state]
        hbm_curve_gib = [peak_gib]  # peak after session 0 (already open above)
        for _ in range(sessions - 1):
            s = engine.reset(conditioning, RolloutParams(horizon=horizon))
            _sync()
            states.append(s)
            hbm_curve_gib.append(torch.cuda.max_memory_allocated() / 1024**3 if on_gpu else 0.0)
        marginal_gib = [
            round(hbm_curve_gib[0] - baseline_gib, 2),
            *[round(hbm_curve_gib[i] - hbm_curve_gib[i - 1], 2) for i in range(1, len(states))],
        ]

        # Round-robin: 1 untimed warmup pass, then >=3 timed passes per session.
        def _round_robin(rounds: int, *, timed: bool) -> list[list[float]]:
            per_session_ms: list[list[float]] = [[] for _ in states]
            for _ in range(rounds):
                for i, s in enumerate(states):
                    _sync()
                    t0 = time.perf_counter()
                    s, _ = engine.step(s, rand_action())
                    _sync()
                    states[i] = s
                    if timed:
                        per_session_ms[i].append((time.perf_counter() - t0) * 1000)
                    if not torch.isfinite(s.context).all():
                        raise AssertionError(
                            f"non-finite WorldState.context at session {i} "
                            f"({sessions} resident) — NaN/inf under N-session concurrency"
                        )
            return per_session_ms

        _round_robin(1, timed=False)
        per_session_step_ms = _round_robin(3, timed=True)

        concurrency = {
            "sessions_requested": sessions,
            "hbm_curve_gib": [round(x, 2) for x in hbm_curve_gib],
            "marginal_gib_measured_per_session": marginal_gib,
            "avg_marginal_gib_measured": round(sum(marginal_gib) / len(marginal_gib), 2),
            "single_session_extrapolated_marginal_gib": round(session_gib, 2),
            "resident_sessions_extrapolated": resident_sessions,
            "resident_sessions_measured": sessions,
            "step_ms_per_session_at_n_resident": [
                round(sum(ms) / len(ms), 2) if ms else None for ms in per_session_step_ms
            ],
            "all_finite": True,
            "round_robin_rounds": 3,
        }

    info = engine.info()
    result = {
        "mode": "control_loop_bench_v0",
        "model": info.model_name,
        "device": info.device,
        "dtype": info.dtype,
        "load_seconds": round(load_s, 1),
        "reset_seconds": round(reset_s, 3),
        "step_ms_warm": round(step_s * 1000, 2),
        "steps_per_sec": round(1 / step_s, 1) if step_s > 0 else None,
        "plan_seconds": round(plan_s, 3),
        "planning_decisions_per_sec": round(1 / plan_s, 4) if plan_s > 0 else None,
        "plan_horizon": horizon,
        "cem": {"samples": samples, "iters": iters} if samples else None,
        "energy_evals_per_plan": energy_evals,
        "energy_evals_per_sec": (
            round(energy_evals / plan_s, 1) if energy_evals and plan_s > 0 else None
        ),
        "weights_gib": round(baseline_gib, 2),
        "session_marginal_gib": round(session_gib, 2),
        "hbm_total_gib": round(hbm_total_gib, 1),
        "resident_sessions_per_gpu": resident_sessions,
        "step_iters": step_iters,
        "plan_calls": plan_calls,
        "state_carryover": True,
        "concurrency": concurrency,
    }
    if extra:
        result.update(extra)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--engine", choices=["vjepa2-ac", "lingbot-va", "dreamzero"], default="vjepa2-ac"
    )
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    ap.add_argument(
        "--fake", action="store_true", help="toy encoder/predictor (CPU/CI, no weights)"
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--step-iters", type=int, default=20)
    ap.add_argument("--plan-calls", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--plan-samples", type=int, default=64)
    ap.add_argument("--plan-cem-iters", type=int, default=3)
    ap.add_argument(
        "--obs-dir",
        default=None,
        help="lingbot-va/dreamzero: dir with seed camera images",
    )
    ap.add_argument(
        "--prompt",
        default="Pick the green cube and place it inside the blue box",
        help="lingbot-va/dreamzero: the goal instruction (text-conditioned policy)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="open N sessions and round-robin step() across them (N>1: true "
        "measured resident-session curve alongside the single-session extrapolation)",
    )
    # LingBot-VA serving-latency levers (scripts/bench_lingbot_va_levers.py's
    # rungs), plumbed here so bench_control_loop.py can drive any of them
    # too. Defaults match LingBotVAConfig's, so existing invocations are
    # unaffected.
    ap.add_argument("--guidance-scale", type=float, default=5.0, help="lingbot-va: video CFG scale")
    ap.add_argument(
        "--action-guidance-scale", type=float, default=1.0, help="lingbot-va: action CFG scale"
    )
    ap.add_argument("--video-steps", type=int, default=5, help="lingbot-va: video denoise steps")
    ap.add_argument("--action-steps", type=int, default=10, help="lingbot-va: action denoise steps")
    ap.add_argument(
        "--attn-mode",
        choices=["torch", "flashattn", "flex"],
        default="torch",
        help="lingbot-va: attention path ('flashattn' is H100-only, not portable)",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="lingbot-va/dreamzero: torch.compile the transformer at load time",
    )
    # DreamZero serving-latency levers (docs/DREAMZERO_PORT_PLAN.md §4),
    # plumbed the same way as the LingBot-VA levers above.
    ap.add_argument("--dz-cfg-scale", type=float, default=5.0, help="dreamzero: CFG scale")
    ap.add_argument(
        "--num-dit-steps",
        type=int,
        default=16,
        help="dreamzero: DiT calls actually run per 16-step denoise loop "
        "(the reference's own undisclosed default is 8 -- pass 8 to "
        "reproduce their path, 16 for the true full-compute baseline)",
    )
    ap.add_argument(
        "--dit-cache-dynamic",
        action="store_true",
        help="dreamzero: enable the cosine-similarity dynamic DiT-cache skip schedule",
    )
    ap.add_argument(
        "--cfg-batched",
        action="store_true",
        help="dreamzero: batch cond+uncond into one forward (falls back to "
        "sequential with a warning until GPU-verified)",
    )
    ap.add_argument(
        "--attn-window",
        type=int,
        default=None,
        help="dreamzero: override local_attn_size (KV window, in frames)",
    )
    args = ap.parse_args()

    if args.engine in ("lingbot-va", "dreamzero") and not args.obs_dir:
        ap.error(f"--engine {args.engine} requires --obs-dir")

    engine = build_engine(args)
    print(f"[clb] engine={args.engine} fake={args.fake}", flush=True)
    extra: dict[str, Any] = {}
    if overrides := lingbot_va_lever_overrides(args):
        extra["lingbot_va_non_default_levers"] = overrides
    if overrides := dreamzero_lever_overrides(args):
        extra["dreamzero_non_default_levers"] = overrides
    result = bench_control_loop(
        engine,
        warmup=args.warmup,
        step_iters=args.step_iters,
        plan_calls=args.plan_calls,
        horizon=args.horizon,
        action_dim=args.action_dim,
        obs_dir=args.obs_dir,
        seed=args.seed,
        sessions=args.sessions,
        extra=extra or None,
    )
    print("[clb] RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
