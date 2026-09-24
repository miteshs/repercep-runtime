"""Tests for the Cosmos 3 Nano engine's model-agnostic chunk-loop layer.

Exercises Protocol conformance, the step/plan mode split, pixel-space frame
chaining, action validation and denormalization, and the conditioning-canvas
geometry. Mirrors ``test_dreamzero.py``: torch-dependent parts skip cleanly
without torch, and the model-specific pipeline is GPU-only (Phase 1, not
landed — ``load()`` is asserted here to raise the port recipe).

The distinctive thing under test versus the two prior interactive ports is
what is *absent*: Cosmos 3 carries no cross-chunk cache (see
``docs/COSMOS3_PORT_PLAN.md`` §2.1), so there is no session bookkeeping, no
``release()``, and — the property worth a regression test —
:meth:`Cosmos3Engine.step` genuinely does not mutate the state it was given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from repercep.models.cosmos3 import (
    FORWARD_DYNAMICS_MODE,
    POLICY_MODE,
    Cosmos3Config,
    Cosmos3Engine,
)
from repercep.runtime.interactive import InteractiveWorldModel
from repercep.runtime.types import Action, ConditioningInput, RolloutParams

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend


class _NamedBackend:
    """Minimal stand-in carrying just the attribute the engine reads (``name``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakePipeline:
    """Deterministic fake of the one stateless call the real pipeline makes.

    ``infer_chunk`` returns frames whose pixel values encode the seed it was
    called with (so chaining is observable), and — in policy mode only — an
    action chunk in normalized ``[-1, 1]`` space. Records every call so the
    step/plan mode split can be asserted.
    """

    def __init__(self, frames_per_chunk: int = 17, action_dim: int = 10) -> None:
        self.frames_per_chunk = frames_per_chunk
        self.action_dim = action_dim
        self.calls: list[dict[str, object]] = []
        self.resolve_calls: list[ConditioningInput] = []

    def resolve_conditioning(self, conditioning: ConditioningInput) -> torch.Tensor:
        import torch

        self.resolve_calls.append(conditioning)
        return torch.zeros(4, 4, 3, dtype=torch.uint8)

    def infer_chunk(
        self,
        *,
        conditioning: torch.Tensor,
        prompt: str,
        mode: str,
        raw_actions: torch.Tensor | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        import torch

        self.calls.append(
            {
                "mode": mode,
                "prompt": prompt,
                "seed": seed,
                "conditioning_sum": int(conditioning.sum().item()),
                "raw_actions_shape": None if raw_actions is None else tuple(raw_actions.shape),
            }
        )
        # Frame i of the chunk carries value (seed + 1) * (i + 1), so the last
        # frame is distinguishable and chaining is visible in the next call's
        # ``conditioning_sum``.
        frames = torch.stack(
            [
                torch.full((4, 4, 3), (seed + 1) * (index + 1), dtype=torch.uint8)
                for index in range(self.frames_per_chunk)
            ]
        )
        if mode != POLICY_MODE:
            return frames, None
        # Normalized action chunk: all -1.0 so denormalization maps it to q01
        # exactly, making the formula checkable by inspection.
        return frames, torch.full((16, self.action_dim), -1.0)


def _toy_engine(
    pipeline: _FakePipeline | None = None,
    *,
    config: Cosmos3Config | None = None,
    with_stats: bool = False,
) -> Cosmos3Engine:
    stats = None
    if with_stats:
        import torch

        # q01 = 0.0, q99 = 2.0 per channel -> denorm(-1) == 0.0, denorm(1) == 2.0.
        stats = (torch.zeros(10), torch.full((10,), 2.0))
    return Cosmos3Engine(
        cast("Backend", _NamedBackend("fake")),
        config if config is not None else Cosmos3Config(prompt="pick up the mug"),
        pipeline=pipeline,
        action_stats=stats,
    )


# --- Protocol conformance / readiness (no torch) ---


def test_engine_satisfies_interactive_protocol() -> None:
    assert issubclass(Cosmos3Engine, InteractiveWorldModel)


def test_engine_not_ready_without_pipeline() -> None:
    assert _toy_engine(pipeline=None).info().ready is False


def test_load_is_phase1_and_carries_the_port_recipe() -> None:
    with pytest.raises(NotImplementedError, match="COSMOS3_PORT_PLAN"):
        _toy_engine(pipeline=None).load()


def test_config_defaults_are_the_policy_notebook_not_the_fd_notebook() -> None:
    """The two notebooks disagree on ``flow_shift`` (policy 5.0, fd 10.0) and
    the cookbook README documents only the fd value — the more discoverable of
    the two. A default of 10.0 here would be a silently wrong sampling
    trajectory, so pin the policy values (port plan §1.2).
    """
    cfg = Cosmos3Config()
    assert cfg.flow_shift == 5.0
    assert cfg.num_inference_steps == 30
    assert cfg.guidance_scale == 1.0
    assert cfg.domain_name == "droid_lerobot"  # plain "droid" is not a valid domain
    assert cfg.chunk_size == 16
    assert cfg.action_dim == 10  # end-effector, NOT DreamZero-DROID's 8D joint-space


def test_engine_has_no_session_machinery() -> None:
    """Cosmos 3 is stateless per chunk (port plan §2.1), so unlike LingBot-VA
    and DreamZero this engine holds no per-session state and exposes no
    ``release()``. If a future change adds one, that is a signal the model was
    misread — or that the port drifted into dressing a stateless call in
    session machinery, which §6 of the plan explicitly warns against.
    """
    assert not hasattr(Cosmos3Engine, "release")
    assert not hasattr(_toy_engine(_FakePipeline()), "_sessions")


# --- the chunk loop (needs torch) ---


def test_reset_delegates_conditioning_to_the_pipeline() -> None:
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    conditioning = ConditioningInput()

    state = engine.reset(conditioning, RolloutParams())

    assert state.step_index == 0
    assert tuple(state.context.shape) == (4, 4, 3)
    assert pipeline.resolve_calls == [conditioning]
    # A reset makes no pipeline *inference* call — for a stateless model there
    # is nothing to prime.
    assert pipeline.calls == []


def test_step_uses_forward_dynamics_and_passes_the_executed_chunk() -> None:
    """``step`` is action-conditioned only if the executed chunk actually
    reaches the model — which means forward-dynamics mode, not policy mode.
    """
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())

    executed = Action(values=[0.0] * (2 * 10), space="cosmos3_chunk_droid_ee")
    nxt, step = engine.step(state, executed)

    assert pipeline.calls[0]["mode"] == FORWARD_DYNAMICS_MODE
    assert pipeline.calls[0]["raw_actions_shape"] == (2, 10)
    assert step.step_index == 1
    assert nxt.step_index == 1
    assert nxt.session_id == state.session_id


def test_step_chains_on_the_last_generated_frame() -> None:
    """The reference ``run_rollout`` feeds each chunk's *last* frame into the
    next call as conditioning (port plan §2.1). With the fake's encoding, the
    last frame of chunk 0 (seed 0) is ``1 * 17 = 17`` in every pixel.
    """
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())
    executed = Action(values=[0.0] * 10, space="cosmos3_chunk_droid_ee")

    first, _ = engine.step(state, executed)
    assert int(first.context.max().item()) == 17
    assert int(first.context.min().item()) == 17

    engine.step(first, executed)
    # Chunk 1 saw chunk 0's last frame: 17 across a 4x4x3 canvas.
    assert pipeline.calls[1]["conditioning_sum"] == 17 * 4 * 4 * 3


def test_step_varies_the_seed_per_chunk() -> None:
    """The reference rollout passes ``seed=chunk_index``; a fixed seed across
    chunks correlates the noise draws and is not what it measures.
    """
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())
    executed = Action(values=[0.0] * 10, space="cosmos3_chunk_droid_ee")

    state, _ = engine.step(state, executed)
    engine.step(state, executed)

    assert [call["seed"] for call in pipeline.calls] == [0, 1]


def test_step_does_not_mutate_the_state_it_was_given() -> None:
    """The seam promises the previous state is not mutated so a planner can
    branch from a shared prefix. LingBot-VA and DreamZero both have to break
    that promise (in-place KV caches); Cosmos 3 is the first port where it
    holds, so it is worth a regression test rather than a comment.
    """
    pytest.importorskip("torch")
    engine = _toy_engine(_FakePipeline())
    state = engine.reset(ConditioningInput(), RolloutParams())
    before = state.context.clone()
    executed = Action(values=[0.0] * 10, space="cosmos3_chunk_droid_ee")

    branch_a, _ = engine.step(state, executed)
    branch_b, _ = engine.step(state, executed)

    assert bool((state.context == before).all())
    assert state.step_index == 0
    # Both branches advanced from the same prefix and agree — the point of the
    # guarantee.
    assert bool((branch_a.context == branch_b.context).all())


def test_step_rejects_a_chunk_that_is_not_a_multiple_of_action_dim() -> None:
    pytest.importorskip("torch")
    engine = _toy_engine(_FakePipeline())
    state = engine.reset(ConditioningInput(), RolloutParams())

    with pytest.raises(ValueError, match="action_dim=10"):
        engine.step(state, Action(values=[0.0] * 7, space="cosmos3_chunk_droid_ee"))


def test_plan_uses_policy_mode_and_denormalizes() -> None:
    pytest.importorskip("torch")
    import torch

    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline, with_stats=True)
    state = engine.reset(ConditioningInput(), RolloutParams())

    action = engine.plan(state, torch.zeros(1), horizon=16)

    assert pipeline.calls[0]["mode"] == POLICY_MODE
    assert pipeline.calls[0]["raw_actions_shape"] is None
    assert action.space == "cosmos3_chunk_droid_ee"
    assert len(action.values) == 10
    # Fake returns -1.0 normalized; with q01=0, q99=2 that denormalizes to q01.
    assert action.values == [0.0] * 10


def test_plan_refuses_to_return_normalized_values_as_meters() -> None:
    """Without quantile stats the honest failure is loud. Returning normalized
    values would produce a plausible-looking, entirely wrong trajectory — the
    worst kind of bug in this codebase's history (see the port plan's
    denormalization notes).
    """
    pytest.importorskip("torch")
    import torch

    engine = _toy_engine(_FakePipeline(), with_stats=False)
    state = engine.reset(ConditioningInput(), RolloutParams())

    with pytest.raises(RuntimeError, match="quantile stats"):
        engine.plan(state, torch.zeros(1), horizon=16)


def test_step_mode_can_fall_back_to_policy() -> None:
    """Whether the post-trained Policy-DROID checkpoint still serves forward
    dynamics is unverified (Phase-1 gate). If it does not, ``step_mode``
    switches to policy — and the executed action stops reaching the model,
    which this test pins so the semantic loss is visible rather than silent.
    """
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline, config=Cosmos3Config(step_mode=POLICY_MODE))
    state = engine.reset(ConditioningInput(), RolloutParams())

    engine.step(state, Action(values=[1.0] * 10, space="cosmos3_chunk_droid_ee"))

    assert pipeline.calls[0]["mode"] == POLICY_MODE
    assert pipeline.calls[0]["raw_actions_shape"] is None  # executed action ignored


# --- conditioning canvas geometry ---


def test_conditioning_canvas_matches_the_reference_layout() -> None:
    """``build_concat_frame``: wrist across the full top half, exterior_1 and
    exterior_2 side by side across the bottom half, on a 640x540 canvas.
    Getting this wrong degrades every downstream number silently, which is
    why it is tested on CPU rather than discovered on a pod.
    """
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    import torch

    engine = _toy_engine(_FakePipeline())
    canvas = engine.build_conditioning_canvas(
        wrist=torch.full((60, 80, 3), 10, dtype=torch.uint8),
        exterior_1=torch.full((60, 80, 3), 20, dtype=torch.uint8),
        exterior_2=torch.full((60, 80, 3), 30, dtype=torch.uint8),
    )

    assert tuple(canvas.shape) == (540, 640, 3)
    # Top half is wrist; bottom half splits at x=320.
    assert int(canvas[100, 320, 0].item()) == 10
    assert int(canvas[400, 100, 0].item()) == 20
    assert int(canvas[400, 500, 0].item()) == 30


def test_conditioning_canvas_honours_configured_dimensions() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    import torch

    engine = _toy_engine(
        _FakePipeline(), config=Cosmos3Config(canvas_width=320, canvas_height=270)
    )
    canvas = engine.build_conditioning_canvas(
        wrist=torch.zeros(60, 80, 3, dtype=torch.uint8),
        exterior_1=torch.zeros(60, 80, 3, dtype=torch.uint8),
        exterior_2=torch.zeros(60, 80, 3, dtype=torch.uint8),
    )
    assert tuple(canvas.shape) == (270, 320, 3)


# --- the shared quantile denormalizer ---


def test_denormalize_quantile_inverts_the_reference_formula() -> None:
    pytest.importorskip("torch")
    import torch

    from repercep.models.action_norm import denormalize_quantile

    q01 = torch.tensor([0.0, -1.0])
    q99 = torch.tensor([2.0, 1.0])
    actions = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])

    out = denormalize_quantile(actions, q01, q99)

    assert torch.allclose(out, torch.tensor([[0.0, -1.0], [2.0, 1.0], [1.0, 0.0]]))


def test_denormalize_quantile_eps_is_explicit() -> None:
    """LingBot-VA's lineage adds 1e-6 to the span and DreamZero's does not.
    The helper takes ``eps`` rather than picking a winner, because a silent
    mismatch produces a plausible, wrong trajectory rather than an error.
    """
    pytest.importorskip("torch")
    import torch

    from repercep.models.action_norm import denormalize_quantile

    actions = torch.tensor([[1.0]])
    q01, q99 = torch.tensor([0.0]), torch.tensor([2.0])

    assert denormalize_quantile(actions, q01, q99, eps=0.0).item() == pytest.approx(2.0)
    assert denormalize_quantile(actions, q01, q99, eps=1e-6).item() == pytest.approx(2.000001)


def test_denormalize_quantile_rejects_channel_mismatch() -> None:
    pytest.importorskip("torch")
    import torch

    from repercep.models.action_norm import denormalize_quantile

    with pytest.raises(ValueError, match="channels"):
        denormalize_quantile(torch.zeros(4, 10), torch.zeros(8), torch.ones(8))
