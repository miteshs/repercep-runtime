"""Tests for the VLA action-token engine's model-agnostic action-loop layer.

Exercises Protocol conformance, session bookkeeping, and — with an injected
fake pipeline — the executed-action->advance->decode step loop and the
candidate-batched plan (the CEM-batching lever that is the differentiator).
Mirrors ``test_lingbot_va.py``: torch-dependent parts skip cleanly without
torch; the model-specific pipeline is a Phase-1 GPU port asserted to raise
(see ``docs/VLA_PORT_PLAN.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from repercep.models.vla import VLAConfig, VLAEngine
from repercep.runtime.interactive import InteractiveWorldModel
from repercep.runtime.types import (
    Action,
    ConditioningInput,
    ResetRequest,
    RolloutParams,
    WorldState,
)

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend


class _NamedBackend:
    """Minimal stand-in carrying just the attribute the engine reads (``name``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeVLAPipeline:
    """Deterministic fake of a token-VLA session pipeline.

    ``decode_action_chunk`` tags candidate ``i`` by writing ``i`` into its first
    action dim, and records every call's ``n_candidates`` so tests can assert
    the batched vs per-candidate path. ``append_executed`` advances the context
    by 1.0 so the clock is observable. ``score_candidates`` puts the minimum at
    ``best_index`` (via that tag), so a correct ``plan`` returns candidate
    ``best_index``.
    """

    def __init__(
        self,
        *,
        action_dim: int = 7,
        chunk: int = 1,
        dim: int = 8,
        supports_batch: bool = True,
        best_index: int = 3,
    ) -> None:
        self.action_dim = action_dim
        self.chunk = chunk
        self.dim = dim
        self.supports_batch = supports_batch
        self.best_index = best_index
        self.reset_calls: list[tuple[str, str | None]] = []
        self.encode_calls: list[str] = []
        self.decode_calls: list[tuple[str, int]] = []
        self.append_calls: list[tuple[str, tuple[int, ...]]] = []
        self.score_calls: list[str] = []
        self.close_calls: list[str] = []
        self.session_state: dict[str, dict[str, object]] = {}

    def reset(self, session_id: str, prompt: str | None) -> None:
        self.reset_calls.append((session_id, prompt))
        self.session_state[session_id] = {"prompt": prompt}

    def encode_observation(self, session_id: str, conditioning: ConditioningInput) -> torch.Tensor:
        import torch

        self.encode_calls.append(session_id)
        # Distinct value per call so two sessions' contexts never alias.
        ctx = torch.full((1, self.dim), float(len(self.encode_calls) - 1))
        self.session_state[session_id]["context"] = ctx
        return ctx

    def decode_action_chunk(
        self, session_id: str, context: torch.Tensor, n_candidates: int
    ) -> torch.Tensor:
        import torch

        self.decode_calls.append((session_id, n_candidates))
        out = torch.zeros(n_candidates, self.chunk, self.action_dim)
        for i in range(n_candidates):
            out[i, 0, 0] = float(i)
        return out

    def append_executed(
        self, session_id: str, executed: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        self.append_calls.append((session_id, tuple(executed.shape)))
        return context + 1.0

    def score_candidates(
        self, session_id: str, candidates: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        self.score_calls.append(session_id)
        return (candidates[:, 0, 0] - float(self.best_index)) ** 2

    def close(self, session_id: str) -> None:
        self.close_calls.append(session_id)


def _toy_engine(
    pipeline: _FakeVLAPipeline | None = None, *, config: VLAConfig | None = None
) -> VLAEngine:
    return VLAEngine(
        cast("Backend", _NamedBackend("fake")),
        config if config is not None else VLAConfig(prompt="pick the green cube", action_dim=7),
        pipeline=pipeline,
    )


# --- Protocol conformance / readiness (no torch) ---


def test_engine_satisfies_interactive_protocol() -> None:
    assert issubclass(VLAEngine, InteractiveWorldModel)


def test_engine_not_ready_without_pipeline() -> None:
    assert _toy_engine(pipeline=None).info().ready is False


def test_info_reports_model_and_backend() -> None:
    info = _toy_engine(_FakeVLAPipeline()).info()
    assert info.model_name == "vla"
    assert info.backend == "fake"
    assert info.ready is True


def test_load_requires_port() -> None:
    # Without the Phase-1 pipeline the real build raises with the port recipe.
    with pytest.raises(RuntimeError, match="VLA_PORT_PLAN"):
        _toy_engine(pipeline=None).load()


# --- the action loop (needs torch) ---


def test_reset_opens_session_with_prompt() -> None:
    pytest.importorskip("torch")
    pipeline = _FakeVLAPipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())
    assert state.step_index == 0
    assert tuple(state.context.shape) == (1, 8)
    assert pipeline.reset_calls == [(state.session_id, "pick the green cube")]
    assert pipeline.encode_calls == [state.session_id]


def test_step_advances_context_and_decodes_next() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakeVLAPipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())

    executed = Action(values=[0.0] * 7, space="vla_7d")
    nxt, step = engine.step(state, executed)

    # Executed chunk shaped (k, action_dim) and pushed before decoding.
    assert pipeline.append_calls == [(state.session_id, (1, 7))]
    # The greedy next chunk was decoded with a single candidate.
    assert pipeline.decode_calls == [(state.session_id, 1)]
    # Context advanced by one step; the incoming envelope is untouched.
    assert step.step_index == 1 and nxt.step_index == 1
    assert torch.all(nxt.context == 1.0)
    assert state.step_index == 0 and torch.all(state.context == 0.0)


def test_plan_batches_all_candidates_in_one_forward() -> None:
    """The differentiator: N candidates decoded in ONE call, best by argmin."""
    torch = pytest.importorskip("torch")
    pipeline = _FakeVLAPipeline(best_index=3)
    engine = _toy_engine(pipeline, config=VLAConfig(action_dim=7, plan_candidates=8))
    state = engine.reset(ConditioningInput(), RolloutParams())

    action = engine.plan(state, torch.zeros(8), horizon=1)

    # One batched decode for all 8 candidates (shared prefix KV) — not 8 calls.
    assert pipeline.decode_calls == [(state.session_id, 8)]
    assert pipeline.score_calls == [state.session_id]
    # argmin over the scores selects candidate 3, whose first action dim is 3.0.
    assert action.space == "vla_7d"
    assert len(action.values) == 7
    assert action.values[0] == 3.0


def test_plan_falls_back_to_per_candidate_loop_without_batch_support() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakeVLAPipeline(supports_batch=False)
    engine = _toy_engine(pipeline, config=VLAConfig(action_dim=7, plan_candidates=5))
    state = engine.reset(ConditioningInput(), RolloutParams())

    action = engine.plan(state, torch.zeros(8), horizon=1)

    # Five separate single-candidate decodes instead of one batched forward.
    assert pipeline.decode_calls == [(state.session_id, 1)] * 5
    assert len(action.values) == 7


def test_plan_batched_disabled_by_config_uses_loop() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakeVLAPipeline(supports_batch=True)
    engine = _toy_engine(
        pipeline, config=VLAConfig(action_dim=7, plan_candidates=4, plan_batched=False)
    )
    state = engine.reset(ConditioningInput(), RolloutParams())
    engine.plan(state, torch.zeros(8), horizon=1)
    assert pipeline.decode_calls == [(state.session_id, 1)] * 4


def test_custom_wire_space_tag() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakeVLAPipeline()
    engine = _toy_engine(pipeline, config=VLAConfig(action_dim=7, space="openvla_7d"))
    state = engine.reset(ConditioningInput(), RolloutParams())
    action = engine.plan(state, torch.zeros(8), horizon=1)
    assert action.space == "openvla_7d"


def test_release_frees_both_layers() -> None:
    pytest.importorskip("torch")
    pipeline = _FakeVLAPipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())
    assert state.session_id in engine._sessions

    engine.release(state)

    assert state.session_id not in engine._sessions
    assert pipeline.close_calls == [state.session_id]

    # Releasing before a pipeline ever loaded (e.g. a WS that drops early) is a no-op.
    _toy_engine(pipeline=None).release(state)


def test_step_rejects_misshapen_action_chunk() -> None:
    pytest.importorskip("torch")
    engine = _toy_engine(_FakeVLAPipeline())
    state = engine.reset(ConditioningInput(), RolloutParams())
    with pytest.raises(ValueError, match="multiple of action_dim"):
        engine.step(state, Action(values=[0.0] * 8))  # 8 is not a multiple of 7


def test_step_rejects_foreign_world_state() -> None:
    torch = pytest.importorskip("torch")
    engine = _toy_engine(_FakeVLAPipeline())
    engine.reset(ConditioningInput(), RolloutParams())
    foreign = WorldState(context=torch.zeros(1, 8), step_index=0, session_id="nope")
    with pytest.raises(KeyError, match="unknown session"):
        engine.step(foreign, Action(values=[0.0] * 7))


def test_two_sessions_keep_distinct_contexts() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakeVLAPipeline()
    engine = _toy_engine(pipeline)

    state_a = engine.reset(ConditioningInput(), RolloutParams())
    state_b = engine.reset(ConditioningInput(), RolloutParams())

    assert state_a.session_id != state_b.session_id
    assert not torch.equal(state_a.context, state_b.context)


# --- serving integration (Phase-3 proof: zero serving changes needed) ---


def test_vla_engine_serves_over_world_session() -> None:
    """A VLAEngine drives ``/v2/world/session`` unchanged — the seam is shared.

    The port plan's Phase-3 claim is that serving a token VLA needs *no* serving
    changes because the WebSocket is already ``interactive_engine``-agnostic.
    This drives the real FastAPI surface end-to-end to prove it.
    """
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from repercep.serving.app import create_app

    engine = _toy_engine(_FakeVLAPipeline())
    with (
        TestClient(create_app(interactive_engine=engine)) as client,
        client.websocket_connect("/v2/world/session") as ws,
    ):
        ws.send_text(ResetRequest().model_dump_json())
        assert ws.receive_json()["step_index"] == 0  # reset acknowledged

        for expected in (1, 2, 3):
            ws.send_text(Action(values=[0.0] * 7).model_dump_json())
            assert ws.receive_json()["step_index"] == expected
