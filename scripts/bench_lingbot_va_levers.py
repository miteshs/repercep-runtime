#!/usr/bin/env python3
"""Serving-latency ladder for the REAL LingBot-VA 2.0 pipeline.

Follows the ``bench_vjepa2_ac_levers.py`` house style (timed stages, warm-up
then measure, one verbatim ``RESULT`` JSON line) but ladders through the
LingBot-VA-specific levers between the measured seam baseline
(``docs/LINGBOT_VA_SEAM_VERIFY.md``, 754.6-801.5 ms/chunk on H100) and the
reference paper's portable-tricks floor (~466 ms — everything below that in
their ladder is CUDA-only: FP8 TensorRT, FlashInfer paged-KV):

- **rung 0 (baseline)**: today's defaults — CFG 5.0/1.0, 5 video / 10 action
  denoise steps, ``attn_mode="torch"`` (SDPA).
- **rung 1 (CFG off)**: ``guidance_scale=1.0`` — halves the DiT forward batch
  AND the KV-cache's batch dim (``LingBotVAPipeline.reset``'s ``use_cfg``
  gate), so this also roughly doubles resident-session density, not just
  latency.
- **rung 2 (step-count scan)**: video steps {4,3,2} and action steps
  {8,6,4}, each independently (holding the other at rung-0's default) —
  layered on rung 1. This is two 1-D scans (6 runs), not the full 3x3 cross
  product, to bound GPU-pod wall time; every combination *tried* is still
  reported (not just the best).
- **rung 3 (torch.compile)**: new ``LingBotVAConfig.compile_transformer``
  flag, wrapping the transformer at construction. Cold (first chunk, compile
  tax included) and warm are reported separately. If the named-KV-cache
  mutation (``cache_name``/``update_cache`` kwargs) causes graph breaks that
  make the number meaningless, this reports "not measurable" rather than a
  fabricated latency.
- **rung 4 (attn_mode)**: ``flex`` (portable — the shipping candidate) and
  ``flashattn`` (H100-only reference point, disclosed as non-portable — it
  will not run on ROCm).

Quality guardrail (disclosed, not oversold): this repo has no offline
task-success metric for LingBot-VA, so the only quality signal available is
per-channel |action| magnitude-distribution consistency (p50, p95) against
rung 0's, at a fixed seed. Consistency is evidence a lever didn't silently
change *what* the model outputs; it is not a correctness or task-success
proof — see ``docs/LINGBOT_VA_SEAM_VERIFY.md``'s "magnitude" methodology
note, which this mirrors.

This is a GPU-pod deliverable: it needs the research repo importable as
``wan_va`` plus the checkpoint bundle (``LingBotVAEngine.load()`` raises a
``RuntimeError`` with the setup recipe otherwise — this script does not catch
or swallow that error for the baseline load, so running it without a GPU
fails immediately and informatively).

    python scripts/bench_lingbot_va_levers.py \
        --obs-dir /workspace/lingbot-va/example/demo \
        --prompt "Pick the green cube and place it inside the blue box"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


def _setup() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))


_setup()

import torch  # noqa: E402

from repercep.backend.registry import select_backend  # noqa: E402
from repercep.models.lingbot_va import LingBotVAConfig, LingBotVAEngine  # noqa: E402
from repercep.runtime.types import (  # noqa: E402
    Action,
    ConditioningInput,
    ConditioningKind,
    RolloutParams,
)

if TYPE_CHECKING:
    from repercep.backend.protocol import Backend
    from repercep.runtime.types import WorldState


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _magnitude_stats(actions: torch.Tensor) -> dict[str, list[float]]:
    """Per-channel ``|action|`` p50/p95 — see the module docstring's guardrail note."""
    if actions.numel() == 0:
        return {"p50": [], "p95": []}
    mag = actions.abs().float()
    p50 = torch.quantile(mag, 0.5, dim=0)
    p95 = torch.quantile(mag, 0.95, dim=0)
    return {"p50": [round(v, 4) for v in p50.tolist()], "p95": [round(v, 4) for v in p95.tolist()]}


def _max_rel_diff(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    diffs = [abs(x - y) / max(abs(y), 1e-6) for x, y in zip(a, b, strict=True)]
    return round(max(diffs), 4)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _load_engine(backend: Backend, **cfg_kwargs: Any) -> tuple[LingBotVAEngine, float]:
    """Construct + load a fresh engine. NOT wrapped in try/except here — the
    ``wan_va`` "not importable"/checkpoint-missing ``RuntimeError`` from
    ``LingBotVAEngine.load()`` must propagate uncaught so this script fails
    loudly (with the setup recipe) rather than silently, exactly as running
    the engine directly would.
    """
    engine = LingBotVAEngine(backend, LingBotVAConfig(**cfg_kwargs))
    t = time.perf_counter()
    engine.load()
    _sync()
    return engine, time.perf_counter() - t


def _open_session(engine: LingBotVAEngine, obs_dir: str, prompt: str) -> WorldState:
    return engine.reset(
        ConditioningInput(kind=ConditioningKind.IMAGE, uri=obs_dir), RolloutParams()
    )


def _run_chunks(
    engine: LingBotVAEngine,
    state: WorldState,
    goal: torch.Tensor,
    *,
    warmup: int,
    measure: int,
) -> tuple[WorldState, list[float], torch.Tensor]:
    """Drive plan→step warmup then measured chunks. Returns the trailing
    state, per-chunk ms (measured chunks only), and the proposed actions
    concatenated across measured chunks (for the magnitude guardrail).
    """
    for _ in range(warmup):
        engine.plan(state, goal, horizon=engine._config.frame_chunk_size)
        proposed = engine._sessions[state.session_id].pending_actions
        assert proposed is not None
        state, _ = engine.step(state, Action(values=proposed.flatten().tolist()))

    chunk_ms: list[float] = []
    action_rows: list[torch.Tensor] = []
    for _ in range(measure):
        engine.plan(state, goal, horizon=engine._config.frame_chunk_size)
        proposed = engine._sessions[state.session_id].pending_actions
        assert proposed is not None
        _sync()
        t = time.perf_counter()
        state, _ = engine.step(state, Action(values=proposed.flatten().tolist()))
        _sync()
        chunk_ms.append((time.perf_counter() - t) * 1000)
        action_rows.append(proposed.detach().float().cpu())
    actions_cat = (
        torch.cat(action_rows, dim=0) if action_rows else torch.zeros(0, 1, dtype=torch.float32)
    )
    return state, chunk_ms, actions_cat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="robbyant/lingbot-va-base", help="HF id or local bundle dir")
    ap.add_argument("--obs-dir", required=True, help="dir with <cam_key>.png seed images")
    ap.add_argument("--prompt", default="Pick the green cube and place it inside the blue box")
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm"], default="auto")
    ap.add_argument("--warmup", type=int, default=2, help="untimed chunks before measuring")
    ap.add_argument("--measure", type=int, default=6, help="timed chunks per rung")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    dev = backend.devices()[0]
    print(f"[levers] backend={backend.name} device={dev.name}", flush=True)

    rungs: dict[str, Any] = {}
    rung0_mag: dict[str, list[float]] | None = None

    def _record(
        name: str,
        chunk_ms: list[float],
        actions_cat: torch.Tensor,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        nonlocal rung0_mag
        mag: dict[str, Any] = _magnitude_stats(actions_cat)
        if rung0_mag is None:
            rung0_mag = {"p50": mag["p50"], "p95": mag["p95"]}
        else:
            mag["p50_max_rel_diff_vs_rung0"] = _max_rel_diff(mag["p50"], rung0_mag["p50"])
            mag["p95_max_rel_diff_vs_rung0"] = _max_rel_diff(mag["p95"], rung0_mag["p95"])
        entry = {
            "chunk_ms": [round(x, 1) for x in chunk_ms],
            "chunk_ms_warm_mean": round(_mean(chunk_ms), 1),
            "action_magnitude": mag,
        }
        if extra:
            entry.update(extra)
        rungs[name] = entry
        print(f"[levers] {name}: warm_mean={entry['chunk_ms_warm_mean']}ms", flush=True)

    goal = torch.zeros(1)  # policy-regime: the prompt is the goal; tensor unused

    # --- rungs 0-2: one load, reused. guidance_scale/action_guidance_scale
    # are read per infer_chunk call and at reset-time (use_cfg -> KV-cache
    # batch dim); step counts are read per infer_chunk call. None of these
    # are baked into the transformer at construction, so a fresh
    # engine.reset() per rung (which resizes the KV cache for the current
    # use_cfg) is enough — no need to reconstruct the transformer. ---
    engine, load_s = _load_engine(backend, repo=args.repo, prompt=args.prompt)
    print(f"[levers] load (rungs 0-2, shared): {load_s:.1f}s", flush=True)

    torch.manual_seed(args.seed)
    state = _open_session(engine, args.obs_dir, args.prompt)
    state, chunk_ms, actions = _run_chunks(
        engine, state, goal, warmup=args.warmup, measure=args.measure
    )
    _record("rung0_baseline", chunk_ms, actions)

    engine._config.guidance_scale = 1.0
    torch.manual_seed(args.seed)
    state = _open_session(engine, args.obs_dir, args.prompt)  # fresh reset resizes KV batch dim
    state, chunk_ms, actions = _run_chunks(
        engine, state, goal, warmup=args.warmup, measure=args.measure
    )
    _record("rung1_cfg_off", chunk_ms, actions, extra={"guidance_scale": 1.0})

    for v_steps in (4, 3, 2):
        engine._config.num_inference_steps = v_steps
        engine._config.action_num_inference_steps = 10
        torch.manual_seed(args.seed)
        state = _open_session(engine, args.obs_dir, args.prompt)
        state, chunk_ms, actions = _run_chunks(
            engine, state, goal, warmup=args.warmup, measure=args.measure
        )
        _record(
            f"rung2_video_steps_{v_steps}",
            chunk_ms,
            actions,
            extra={"video_steps": v_steps, "action_steps": 10},
        )
    engine._config.num_inference_steps = 5
    for a_steps in (8, 6, 4):
        engine._config.action_num_inference_steps = a_steps
        torch.manual_seed(args.seed)
        state = _open_session(engine, args.obs_dir, args.prompt)
        state, chunk_ms, actions = _run_chunks(
            engine, state, goal, warmup=args.warmup, measure=args.measure
        )
        _record(
            f"rung2_action_steps_{a_steps}",
            chunk_ms,
            actions,
            extra={"video_steps": 5, "action_steps": a_steps},
        )
    engine._config.action_num_inference_steps = 10

    # --- rung 3: torch.compile — needs a fresh transformer construction. ---
    try:
        compiled_engine, compiled_load_s = _load_engine(
            backend,
            repo=args.repo,
            prompt=args.prompt,
            guidance_scale=1.0,
            compile_transformer=True,
        )
        torch.manual_seed(args.seed)
        state = _open_session(compiled_engine, args.obs_dir, args.prompt)
        # Cold: first measured chunk, compile tax included.
        state, cold_ms, _cold_actions = _run_chunks(
            compiled_engine, state, goal, warmup=0, measure=1
        )
        # Warm: compile already paid for, isolated from rungs 0-2's mean.
        state, warm_ms, warm_actions = _run_chunks(
            compiled_engine, state, goal, warmup=args.warmup, measure=args.measure
        )
        _record(
            "rung3_torch_compile",
            warm_ms,
            warm_actions,
            extra={
                "load_seconds": round(compiled_load_s, 1),
                "cold_first_chunk_ms": round(cold_ms[0], 1) if cold_ms else None,
            },
        )
    except Exception as exc:  # the graph-break/compile failure mode itself
        rungs["rung3_torch_compile"] = {
            "measurable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[levers] rung3 torch.compile: not measurable — {exc}", flush=True)

    # --- rung 4: attn_mode. ``flex`` is the portable shipping candidate;
    # ``flashattn`` is disclosed as an H100-only reference point (it will not
    # run on ROCm — no flash-attn wheel there). ---
    for mode, portable in (("flex", True), ("flashattn", False)):
        try:
            m_engine, m_load_s = _load_engine(
                backend, repo=args.repo, prompt=args.prompt, guidance_scale=1.0, attn_mode=mode
            )
            torch.manual_seed(args.seed)
            state = _open_session(m_engine, args.obs_dir, args.prompt)
            state, chunk_ms, actions = _run_chunks(
                m_engine, state, goal, warmup=args.warmup, measure=args.measure
            )
            _record(
                f"rung4_attn_{mode}",
                chunk_ms,
                actions,
                extra={"portable": portable, "load_seconds": round(m_load_s, 1)},
            )
        except Exception as exc:
            rungs[f"rung4_attn_{mode}"] = {
                "measurable": False,
                "portable": portable,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[levers] rung4 attn={mode}: not measurable — {exc}", flush=True)

    result = {
        "bench": "lingbot_va_levers",
        "device": dev.name,
        "seed": args.seed,
        "warmup_chunks": args.warmup,
        "measured_chunks": args.measure,
        "quality_guardrail_note": (
            "action-magnitude (|action| p50/p95 per channel) vs rung0, at a "
            "fixed seed, is the only quality signal available for LingBot-VA "
            "in this repo — there is no offline task-success metric. "
            "Consistency shows a lever didn't silently change the output "
            "distribution; it does not prove task correctness."
        ),
        "rungs": rungs,
    }
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
