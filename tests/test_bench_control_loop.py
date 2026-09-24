"""Tests for ``scripts/bench_control_loop.py`` — the control-loop leaderboard harness.

Drives the measurement core against the V-JEPA engine with toy weights (the
``--fake`` path), so the four leaderboard metrics are exercised end-to-end on
CPU without downloads. Mirrors ``test_fvd.py``'s load-by-path convention for
script modules.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch

_SRC = Path(__file__).resolve().parent.parent / "scripts" / "bench_control_loop.py"
_spec = importlib.util.spec_from_file_location("bench_control_loop_mod", _SRC)
assert _spec is not None and _spec.loader is not None
_clb = importlib.util.module_from_spec(_spec)
sys.modules["bench_control_loop_mod"] = _clb
_spec.loader.exec_module(_clb)


def _fake_args(**overrides: object) -> object:
    import argparse

    ns = argparse.Namespace(
        engine="vjepa2-ac",
        backend="cpu",
        dtype="fp32",
        fake=True,
        action_dim=4,
        plan_samples=6,
        plan_cem_iters=2,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_bench_control_loop_fake_engine_metrics() -> None:
    pytest.importorskip("torch")
    engine = _clb.build_engine(_fake_args())
    result = _clb.bench_control_loop(
        engine, warmup=1, step_iters=5, plan_calls=1, horizon=2, action_dim=4
    )
    assert result["mode"] == "control_loop_bench_v0"
    assert result["model"] == "vjepa2-ac-300m"
    assert result["step_ms_warm"] > 0
    assert result["planning_decisions_per_sec"] > 0
    # V-JEPA-class engine: energy metrics populated from the CEM config.
    assert result["energy_evals_per_plan"] == 6 * 2
    assert result["energy_evals_per_sec"] > 0
    # CPU run: no HBM story to tell.
    assert result["resident_sessions_per_gpu"] is None
    assert result["state_carryover"] is True


def test_bench_control_loop_result_is_json_serializable() -> None:
    pytest.importorskip("torch")
    import json

    engine = _clb.build_engine(_fake_args())
    result = _clb.bench_control_loop(
        engine, warmup=0, step_iters=2, plan_calls=1, horizon=2, action_dim=4
    )
    assert json.loads(json.dumps(result))["plan_horizon"] == 2


def test_bench_control_loop_sessions_default_is_backward_compatible() -> None:
    """``sessions`` defaults to 1: no concurrency phase runs, and the result
    matches the pre-flag shape (just an added ``concurrency: None`` key).
    """
    pytest.importorskip("torch")
    engine = _clb.build_engine(_fake_args())
    result = _clb.bench_control_loop(
        engine, warmup=1, step_iters=3, plan_calls=1, horizon=2, action_dim=4
    )
    assert result["concurrency"] is None
    assert result["step_ms_warm"] > 0


def test_bench_control_loop_sessions_n_reports_true_measured_curve() -> None:
    """``--sessions N`` (N>1): opens N sessions, records a true incremental
    HBM curve (not the single-session extrapolation), round-robins >=3 step()
    calls per session, and asserts finiteness — the Part 2a concurrency
    hardening. Exercised generically through ``--fake`` (CPU has no HBM
    story, but the round-robin/finite-assertion/session-bookkeeping logic is
    identical to the GPU path).
    """
    pytest.importorskip("torch")
    engine = _clb.build_engine(_fake_args())
    result = _clb.bench_control_loop(
        engine, warmup=1, step_iters=3, plan_calls=1, horizon=2, action_dim=4, sessions=3
    )
    conc = result["concurrency"]
    assert conc is not None
    assert conc["sessions_requested"] == 3
    assert conc["resident_sessions_measured"] == 3
    assert len(conc["hbm_curve_gib"]) == 3
    assert len(conc["marginal_gib_measured_per_session"]) == 3
    assert len(conc["step_ms_per_session_at_n_resident"]) == 3
    assert conc["round_robin_rounds"] == 3
    assert conc["all_finite"] is True
    # Every per-session step latency was actually measured (>=3 timed rounds).
    assert all(ms is not None and ms >= 0 for ms in conc["step_ms_per_session_at_n_resident"])


def test_bench_control_loop_sessions_raises_on_non_finite_step() -> None:
    """The finite-result assertion actually fires on NaN/inf, rather than
    silently accepting a corrupted rollout.
    """
    pytest.importorskip("torch")
    import dataclasses

    engine = _clb.build_engine(_fake_args())

    real_step = engine.step
    real_reset = engine.reset
    reset_calls = {"n": 0}

    def _counting_reset(conditioning: object, params: object) -> object:
        reset_calls["n"] += 1
        return real_reset(conditioning, params)

    def _poison_step(state: object, action: object) -> object:
        new_state, latent_step = real_step(state, action)
        # Only the concurrency phase opens a 2nd+ session (main flow resets
        # once) — poisoning gates on that instead of a magic step() call
        # count, so it doesn't depend on plan()'s internal CEM rollout count.
        if reset_calls["n"] > 1:
            ctx = new_state.context.clone()
            ctx[0, 0] = float("nan")
            new_state = dataclasses.replace(new_state, context=ctx)
        return new_state, latent_step

    engine.step = _poison_step
    engine.reset = _counting_reset
    with pytest.raises(AssertionError, match="non-finite"):
        _clb.bench_control_loop(
            engine, warmup=0, step_iters=1, plan_calls=1, horizon=2, action_dim=4, sessions=2
        )


def test_lingbot_va_lever_overrides_only_discloses_non_default() -> None:
    """RESULT-line disclosure (CONTROL_LOOP_BENCH.md §4): only flags that
    differ from ``LingBotVAConfig``'s defaults appear.
    """
    args = _fake_args(
        engine="lingbot-va",
        guidance_scale=1.0,  # non-default (CFG off)
        action_guidance_scale=1.0,  # default
        video_steps=5,  # default
        action_steps=4,  # non-default
        attn_mode="flex",  # non-default
        compile=False,  # default
    )
    overrides = _clb.lingbot_va_lever_overrides(args)
    assert overrides == {"guidance_scale": 1.0, "action_steps": 4, "attn_mode": "flex"}

    # vjepa2-ac: never discloses LingBot-VA levers.
    assert _clb.lingbot_va_lever_overrides(_fake_args()) == {}


def test_build_engine_dreamzero_constructs_configured_engine() -> None:
    """``build_engine`` wires every dreamzero CLI lever into ``DreamZeroConfig``
    without needing GPU/weights — construction alone never calls ``load()``.
    """
    pytest.importorskip("torch")
    from repercep.models.dreamzero import DreamZeroEngine

    args = _fake_args(
        engine="dreamzero",
        fake=False,
        prompt="pick up the mug",
        dz_cfg_scale=1.0,
        num_dit_steps=8,
        dit_cache_dynamic=True,
        cfg_batched=True,
        compile=True,
        attn_window=12,
    )
    engine = _clb.build_engine(args)
    assert isinstance(engine, DreamZeroEngine)
    cfg = engine._config
    assert cfg.prompt == "pick up the mug"
    assert cfg.cfg_scale == 1.0
    assert cfg.num_dit_steps == 8
    assert cfg.enable_dit_cache is True
    assert cfg.cfg_batched is True
    assert cfg.compile is True
    assert cfg.local_attn_size == 12


def test_build_engine_dreamzero_fake_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="dreamzero"):
        _clb.build_engine(_fake_args(engine="dreamzero", fake=True))


def test_dreamzero_lever_overrides_only_discloses_non_default() -> None:
    args = _fake_args(
        engine="dreamzero",
        fake=False,
        dz_cfg_scale=1.0,  # non-default
        num_dit_steps=16,  # default
        dit_cache_dynamic=False,  # default
        cfg_batched=True,  # non-default
        compile=False,  # default
        attn_window=None,  # default
    )
    overrides = _clb.dreamzero_lever_overrides(args)
    assert overrides == {"dz_cfg_scale": 1.0, "cfg_batched": True}

    # lingbot-va / vjepa2-ac: never disclose DreamZero levers.
    assert _clb.dreamzero_lever_overrides(_fake_args()) == {}


def test_bench_control_loop_dreamzero_chunk_shape_and_forces_real_plan_call() -> None:
    """Mirrors the LingBot-VA chunked-engine test above, but for DreamZero's
    different chunk-geometry field (``num_action_per_block`` rows directly,
    not ``frame_chunk_size x action_per_frame``) — exercises
    ``_chunk_len``'s dispatch and the shared pending_actions-clearing fix.
    """
    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repercep.models.dreamzero import DreamZeroConfig, DreamZeroEngine
    from test_dreamzero import _FakePipeline, _NamedBackend

    class _CountingPipeline(_FakePipeline):
        def __init__(self, num_frame_per_block: int = 2, action_dim: int = 8, dim: int = 4) -> None:
            super().__init__(
                num_frame_per_block=num_frame_per_block, action_dim=action_dim, dim=dim
            )
            self.infer_chunk_calls = 0

        def infer_chunk(
            self,
            session_id: str,
            current_start_frame: int,
            init_latent: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.infer_chunk_calls += 1
            return super().infer_chunk(session_id, current_start_frame, init_latent)

    pipeline = _CountingPipeline(num_frame_per_block=2, action_dim=8, dim=4)
    engine = DreamZeroEngine(
        _NamedBackend("fake"),  # type: ignore[arg-type]
        DreamZeroConfig(prompt="test", num_action_per_block=3, action_dim=8, used_action_dim=8),
        pipeline=pipeline,
    )

    result = _clb.bench_control_loop(
        engine, warmup=1, step_iters=1, plan_calls=2, horizon=2, obs_dir="unused"
    )
    assert result["state_carryover"] is True
    # The rand_action chunk was sized off num_action_per_block=3, not
    # LingBot's frame_chunk_size x action_per_frame fields (which DreamZero's
    # config doesn't have) -- confirmed via the executed chunk shape pushed
    # to recondition().
    assert pipeline.recondition_calls[0][1] == (3, 8)
    # warmup step + timed step + goal step = 3 step()-driven infer_chunk
    # calls, then one REAL infer_chunk per timed plan() call (pending_actions
    # cleared each time) -- 3 + 2 = 5, same discipline as LingBot-VA's test.
    assert pipeline.infer_chunk_calls == 3 + 2


def test_bench_control_loop_chunked_engine_forces_real_plan_call() -> None:
    """LingBot-VA-shaped (chunked) engines: plan() must do real work, not a
    cache hit off the pending_actions a preceding step() parked (the bug this
    guards: the harness's goal-computation step() would otherwise make every
    timed plan() call a free no-op lookup instead of a real chunk denoise).
    """
    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repercep.models.lingbot_va import LingBotVAConfig, LingBotVAEngine
    from test_lingbot_va import _FakePipeline, _NamedBackend

    class _CountingPipeline(_FakePipeline):
        def __init__(self, chunk: int = 4, action_dim: int = 30, dim: int = 8) -> None:
            super().__init__(chunk=chunk, action_dim=action_dim, dim=dim)
            self.infer_chunk_calls = 0

        def infer_chunk(
            self, session_id: str, frame_st_id: int, init_latent: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.infer_chunk_calls += 1
            return super().infer_chunk(session_id, frame_st_id, init_latent)

    pipeline = _CountingPipeline(chunk=2, action_dim=3, dim=4)
    engine = LingBotVAEngine(
        _NamedBackend("fake"),  # type: ignore[arg-type]
        LingBotVAConfig(prompt="test", frame_chunk_size=2, action_per_frame=1, used_action_dim=3),
        pipeline=pipeline,
    )

    result = _clb.bench_control_loop(
        engine, warmup=1, step_iters=1, plan_calls=2, horizon=2, obs_dir="unused"
    )
    assert result["state_carryover"] is True
    # warmup step + timed step + goal step = 3 step()-driven infer_chunk calls,
    # then one REAL infer_chunk per timed plan() call (pending_actions cleared
    # each time) — 3 + 2 = 5. Without the fix this would be 3 + 1 (or fewer):
    # the goal step's cached proposal would serve every plan() call for free.
    assert pipeline.infer_chunk_calls == 3 + 2
