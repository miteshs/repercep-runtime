"""Tests for the DreamZero engine's model-agnostic chunk-loop layer.

Exercises Protocol conformance, session bookkeeping, and — with an injected
fake pipeline — the recondition-then-predict step loop and policy-mode plan.
Mirrors ``test_lingbot_va.py``: torch-dependent parts skip cleanly without
torch; the model-specific pipeline (:mod:`repercep.models.dreamzero_pipeline`)
is GPU-only and asserted here only to raise the setup recipe absent the
research repo (see ``docs/DREAMZERO_PORT_PLAN.md`` §2b for the GPU-verified
real forward pass this pipeline is built from).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from repercep.models.dreamzero import DreamZeroConfig, DreamZeroEngine
from repercep.runtime.interactive import InteractiveWorldModel
from repercep.runtime.types import Action, ConditioningInput, RolloutParams, WorldState

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend


class _NamedBackend:
    """Minimal stand-in carrying just the attribute the engine reads (``name``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakePipeline:
    """Deterministic fake of the reference server's session surface.

    ``infer_chunk`` returns latents filled with ``current_start_frame`` (so
    tests can see the cache clock advance) and an action chunk whose first
    row is ``[current_start_frame, 0, ...]``; ``recondition`` consumes
    ``num_frame_per_block`` frames per executed action chunk and records
    every call for assertions.
    """

    def __init__(self, num_frame_per_block: int = 1, action_dim: int = 7, dim: int = 8) -> None:
        self.num_frame_per_block = num_frame_per_block
        self.action_dim = action_dim
        self.dim = dim
        self.reset_calls: list[tuple[str, str | None]] = []
        self.recondition_calls: list[tuple[str, tuple[int, ...], int]] = []
        self.close_calls: list[str] = []
        self.encode_calls: list[str] = []
        # Per-session state, keyed explicitly by session_id — mirrors the real
        # pipeline's session dict closely enough to exercise the
        # session-association discipline the ``session_id`` parameter buys
        # (see LingBot-VA's interleaved-reset regression test).
        self.session_state: dict[str, dict[str, object]] = {}

    def reset(self, session_id: str, prompt: str | None) -> None:
        self.reset_calls.append((session_id, prompt))
        self.session_state[session_id] = {"prompt": prompt}

    def close(self, session_id: str) -> None:
        self.close_calls.append(session_id)

    def encode_observation(self, session_id: str, conditioning: ConditioningInput) -> torch.Tensor:
        import torch

        self.encode_calls.append(session_id)
        latent = torch.full((self.num_frame_per_block, self.dim), float(len(self.encode_calls) - 1))
        self.session_state[session_id]["init_latent"] = latent
        return latent

    def infer_chunk(
        self, session_id: str, current_start_frame: int, init_latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        latents = torch.full((self.num_frame_per_block, self.dim), float(current_start_frame))
        actions = torch.zeros(32, self.action_dim)
        actions[0, 0] = float(current_start_frame)
        return latents, actions

    def recondition(
        self,
        session_id: str,
        actions: torch.Tensor,
        obs_latent: torch.Tensor | None,
        current_start_frame: int,
    ) -> int:
        self.recondition_calls.append((session_id, tuple(actions.shape), current_start_frame))
        return self.num_frame_per_block


def _toy_engine(pipeline: _FakePipeline | None = None) -> DreamZeroEngine:
    return DreamZeroEngine(
        cast("Backend", _NamedBackend("fake")),
        # Toy 7-wide action space for test shapes -- override the real
        # DreamZero-DROID default (action_dim=32 padded, used_action_dim=8)
        # so both fields agree at 7, matching the fake pipeline below.
        DreamZeroConfig(prompt="pick up the mug", action_dim=7, used_action_dim=7),
        pipeline=pipeline,
    )


def test_build_pipeline_sets_torchdynamo_disable_before_groot_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test (caught on the 2026-07-14 H100 run): ``config.compile``
    must control ``TORCHDYNAMO_DISABLE`` *before* the research package/model
    is imported/constructed, or the reference's own unconditional
    ``torch.compile`` wrapping of the flow-matching scheduler hits
    ``FailOnRecompileLimitHit`` on the very first chunk (port plan §2b) --
    silently reintroduced when the Phase-2 ``compile`` lever field was added
    without wiring its actual mechanism.
    """
    import os

    monkeypatch.delenv("REPERCEP_DREAMZERO_SRC", raising=False)
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    from repercep.models.dreamzero_pipeline import build_pipeline

    with pytest.raises(RuntimeError, match="DREAMZERO_PORT_PLAN"):
        build_pipeline(cast("Backend", _NamedBackend("fake")), DreamZeroConfig())
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"

    with pytest.raises(RuntimeError, match="DREAMZERO_PORT_PLAN"):
        build_pipeline(cast("Backend", _NamedBackend("fake")), DreamZeroConfig(compile=True))
    assert os.environ["TORCHDYNAMO_DISABLE"] == "0"


def test_config_phase2_lever_defaults_are_the_true_baseline() -> None:
    """Phase-2 lever fields (docs/DREAMZERO_PORT_PLAN.md §4) default to the
    true full-compute baseline, not the reference's silent approximations —
    ``num_dit_steps=16`` (not the reference's undisclosed default of 8) and
    every other lever off/unset.
    """
    cfg = DreamZeroConfig()
    assert cfg.num_dit_steps == 16
    assert cfg.cfg_batched is False
    assert cfg.compile is False
    assert cfg.local_attn_size is None


# --- Protocol conformance / readiness (no torch) ---


def test_engine_satisfies_interactive_protocol() -> None:
    assert issubclass(DreamZeroEngine, InteractiveWorldModel)


def test_engine_not_ready_without_pipeline() -> None:
    engine = _toy_engine(pipeline=None)
    assert engine.info().ready is False


def test_load_requires_research_repo() -> None:
    # Without the research repo importable as 'groot' the real-pipeline
    # build fails with the setup recipe (the GPU box sys.path-inserts it per
    # the port plan) -- same pattern as LingBot-VA's load().
    with pytest.raises(RuntimeError, match="DREAMZERO_PORT_PLAN"):
        _toy_engine(pipeline=None).load()


def test_load_is_noop_when_pipeline_already_injected() -> None:
    # A test (or future advanced caller) that injects a pipeline directly
    # must not be tripped up by load() trying to build a real one.
    engine = _toy_engine(_FakePipeline())
    engine.load()  # must not raise
    assert engine.is_loaded


# --- the chunk loop (needs torch for the latent tensors) ---


def test_reset_opens_session_with_prompt() -> None:
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())
    assert state.step_index == 0
    assert tuple(state.context.shape) == (1, 8)
    assert pipeline.reset_calls == [(state.session_id, "pick up the mug")]


def test_release_closes_session_on_pipeline_and_drops_engine_bookkeeping() -> None:
    """``release()`` frees both layers of per-session state (the leak fix).

    Two layers hold session state: this engine's own ``_sessions`` (frame
    clock, parked action proposal) and the pipeline's session-keyed KV +
    cross-attn caches. Without dropping both, a churn of short-lived sessions
    leaks GPU tensors forever (the LingBot-VA lesson).
    """
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())
    assert state.session_id in engine._sessions

    engine.release(state)

    assert state.session_id not in engine._sessions
    assert pipeline.close_calls == [state.session_id]

    # A session release before the engine ever loaded a pipeline (e.g. a
    # WebSocket that disconnects before its first reset()) must not raise.
    _toy_engine(pipeline=None).release(state)


def test_step_reconditions_then_predicts() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())

    executed = Action(values=[0.0] * (2 * 7), space="dreamzero_chunk_droid")
    nxt, step = engine.step(state, executed)

    # Executed chunk shaped (k, action_dim) and pushed before predicting.
    assert pipeline.recondition_calls == [(state.session_id, (2, 7), 0)]
    # The cache clock advanced by one block; the new context is the new block.
    assert step.step_index == 1 and nxt.step_index == 1
    assert torch.all(nxt.context == 1.0)
    # The previous WorldState object is untouched (single live branch caveat
    # is about the cache, not the envelope).
    assert state.step_index == 0 and torch.all(state.context == 0.0)


def test_plan_returns_models_proposed_action() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())

    # Fresh session: plan predicts a block itself (current_start_frame == 0).
    action = engine.plan(state, torch.zeros(8), horizon=1)
    assert action.space == "dreamzero_chunk_droid"
    assert len(action.values) == 7
    assert action.values[0] == 0.0

    # After a step, plan reuses the block the step already predicted (parked),
    # whose first row encodes the advanced cache clock.
    nxt, _ = engine.step(state, Action(values=[0.0] * 7))
    action2 = engine.plan(nxt, torch.zeros(8), horizon=1)
    assert action2.values[0] == 1.0


def test_step_rejects_misshapen_action_chunk() -> None:
    pytest.importorskip("torch")
    engine = _toy_engine(_FakePipeline())
    state = engine.reset(ConditioningInput(), RolloutParams())
    with pytest.raises(ValueError, match="multiple of action_dim"):
        engine.step(state, Action(values=[0.0] * 8))


def test_step_rejects_foreign_world_state() -> None:
    torch = pytest.importorskip("torch")
    engine = _toy_engine(_FakePipeline())
    engine.reset(ConditioningInput(), RolloutParams())
    foreign = WorldState(context=torch.zeros(1, 8), step_index=0, session_id="nope")
    with pytest.raises(KeyError, match="unknown session"):
        engine.step(foreign, Action(values=[0.0] * 7))


# --- session-ordering (mirrors LingBot-VA's interleaved-reset regression) ---


def test_interleaved_resets_associate_observations_with_correct_session() -> None:
    """Two sessions opened back-to-back before either's observation is
    encoded must not have their latents cross-wired — the explicit
    ``session_id`` parameter on ``encode_observation`` is what guarantees
    this regardless of call order (see LingBot-VA's identical regression
    test for the bug this pattern prevents)."""
    pytest.importorskip("torch")
    import torch

    pipeline = _FakePipeline()
    pipeline.reset("sess-a", "prompt a")
    pipeline.reset("sess-b", "prompt b")  # "B" is now the most-recently-inserted session.
    latent_a = pipeline.encode_observation("sess-a", ConditioningInput())
    latent_b = pipeline.encode_observation("sess-b", ConditioningInput())

    assert pipeline.session_state["sess-a"]["init_latent"] is latent_a
    assert pipeline.session_state["sess-b"]["init_latent"] is latent_b
    assert not torch.equal(latent_a, latent_b)
    assert pipeline.encode_calls == ["sess-a", "sess-b"]


def test_two_engine_sessions_keep_distinct_contexts() -> None:
    """End-to-end (through ``DreamZeroEngine.reset``): two sessions opened on
    one engine get their own, non-aliased ``WorldState.context``."""
    torch = pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)

    state_a = engine.reset(ConditioningInput(), RolloutParams())
    state_b = engine.reset(ConditioningInput(), RolloutParams())

    assert state_a.session_id != state_b.session_id
    assert not torch.equal(state_a.context, state_b.context)
    assert pipeline.session_state[state_a.session_id]["init_latent"] is state_a.context
    assert pipeline.session_state[state_b.session_id]["init_latent"] is state_b.context
