"""Tests for the interactive (action-conditioned) world-model seam.

Exercises the wire contract, Protocol conformance, and — with an injected fake
encoder + predictor — the real rollout and CEM/energy planner. The parts that
build latent tensors are gated on torch and skip cleanly on a box without it,
matching the rest of the suite (accelerator-specific paths skip rather than
fail). The model-specific weight load remains a port and is asserted to raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
from pydantic import ValidationError

from repercep.models.vjepa2_ac import (
    VJepa2ACConfig,
    VJepa2ACEngine,
    _ac_raw_qkv,
    _ac_rotate_augmented,
    _AcPredictorAdapter,
    _infer_tokens_per_frame,
)
from repercep.runtime.engine import EngineInfo
from repercep.runtime.interactive import InteractiveWorldModel
from repercep.runtime.types import (
    Action,
    ConditioningInput,
    LatentStep,
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


class _StubInteractive:
    """A torch-backed stub world model that satisfies ``InteractiveWorldModel``."""

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name="stub", backend="cpu", device="cpu:0", dtype="float32", ready=True
        )

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        import torch

        return WorldState(context=torch.zeros(1, 4), step_index=0, session_id="s0")

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        nxt = WorldState(
            context=state.context,
            step_index=state.step_index + 1,
            session_id=state.session_id,
        )
        return nxt, LatentStep(step_index=nxt.step_index)

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        return Action(values=[0.0])


class _FakeEncoder:
    """Returns a fixed ``(1, T, D)`` feature tensor regardless of input."""

    def __init__(self, context: torch.Tensor) -> None:
        self._context = context

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self._context.unsqueeze(0)


class _FakePredictor:
    """Linear toy dynamics: next state = last context frame + action."""

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return context[-1] + action


def _toy_engine(ctx0: torch.Tensor, *, context_frames: int) -> VJepa2ACEngine:
    return VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(
            action_dim=int(ctx0.shape[-1]),
            context_frames=context_frames,
            seed_frames=2,
            seed_resolution=8,
        ),
        encoder=_FakeEncoder(ctx0),
        predictor=_FakePredictor(),
    )


# --- wire types (no torch) ---


def test_action_requires_at_least_one_value() -> None:
    with pytest.raises(ValidationError):
        Action(values=[])
    assert Action(values=[0.1, -0.2], space="ee_delta").space == "ee_delta"


def test_action_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Action.model_validate({"values": [0.0], "bogus": 1})


def test_rollout_params_bounds() -> None:
    assert RolloutParams().horizon == 16
    with pytest.raises(ValidationError):
        RolloutParams(horizon=0)
    with pytest.raises(ValidationError):
        RolloutParams(horizon=10_000)


def test_reset_request_defaults() -> None:
    req = ResetRequest()
    assert req.conditioning.kind is ConditioningInput().kind
    assert req.params.decode_pixels is False


def test_latent_step_envelope_carries_no_tensor() -> None:
    step = LatentStep(step_index=3, energy=1.5)
    assert '"step_index":3' in step.model_dump_json()
    assert LatentStep(step_index=0).frame is None


# --- Protocol conformance (no torch) ---


def test_engine_satisfies_interactive_protocol() -> None:
    # Method-only Protocol → issubclass works with no instance and no torch.
    assert issubclass(VJepa2ACEngine, InteractiveWorldModel)


def test_stub_is_interactive_instance() -> None:
    assert isinstance(_StubInteractive(), InteractiveWorldModel)


def test_engine_not_ready_without_predictor() -> None:
    # Inject only the encoder; the predictor still needs its (GPU/network) load,
    # so the engine reports not-ready until both halves are present.
    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")), VJepa2ACConfig(), encoder=object()
    )
    assert engine.info().ready is False


def test_infer_tokens_per_frame() -> None:
    # Real encoder: (crop/patch)**2 spatial patch tokens per frame; stub: 1.
    class _Cfg:
        crop_size = 256
        patch_size = 16

    class _Enc:
        config = _Cfg()

    assert _infer_tokens_per_frame(_Enc()) == 256
    assert _infer_tokens_per_frame(object()) == 1


def test_ac_predictor_adapter_returns_next_frame_block() -> None:
    torch = pytest.importorskip("torch")

    class _RawPredictor:
        """Mimics the research predictor: (x, actions, states) -> (B, N, D).

        Asserts the per-frame action/state shape the real predictor requires.
        """

        def __call__(self, x: Any, actions: Any, states: Any) -> Any:
            assert x.shape[0] == 1 and actions.shape == states.shape
            assert actions.shape == (1, 3, 7)  # (B, T frames, action_dim)
            return x + actions.sum()

    adapter = _AcPredictorAdapter(_RawPredictor(), tokens_per_frame=4, action_dim=7)
    context = torch.zeros(12, 8)  # 3 frames x P=4 patch tokens, D=8
    nxt = adapter(context, torch.ones(7))
    assert tuple(nxt.shape) == (4, 8)  # the trailing P rows = predicted next frame


def _build_mini_ac_predictor() -> Any:
    """A miniature two-layer AC predictor: the same augmented-token layout,
    axial RoPE, block-causal attention and residual stack as the real
    V-JEPA 2-AC predictor, batch-agnostic (``forward`` never hardcodes batch
    size, so it exercises both the single-session and CEM-batched cache
    paths). Shared by the growing-window and batched-KV parity tests below.

    A factory function, not a module-level class: the classes subclass
    ``nn.Module``, so defining them at import time would break this file's
    "skip cleanly without torch" contract for every other test in it. Deferred
    inside here, called only from tests that already ran
    ``pytest.importorskip("torch")`` first.
    """
    nn = pytest.importorskip("torch.nn")
    functional = pytest.importorskip("torch.nn.functional")
    import torch

    class _MiniAttention(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self, dim: int = 24, heads: int = 2) -> None:
            super().__init__()
            self.num_heads = heads
            self.head_dim = dim // heads
            self.d_dim = self.h_dim = self.w_dim = 4
            self.grid_size = 1
            self.proj_drop_prob = 0.0
            self.is_causal = False
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Identity()

        def separate_positions(self, ids: Any, height: int, width: int) -> Any:
            frame = ids // (height * width)
            within = ids - frame * height * width
            row = within // width
            return frame.float(), row.float(), (within - row * width).float()

        def forward(
            self,
            x: Any,
            *,
            attn_mask: Any,
            action_tokens: int,
            **kwargs: Any,
        ) -> Any:
            frames = int(kwargs["T"])
            height = int(kwargs["H"])
            width = int(kwargs["W"])
            q, k, v = _ac_raw_qkv(self, x)
            q = _ac_rotate_augmented(self, q, frames, height, width, action_tokens)
            k = _ac_rotate_augmented(self, k, frames, height, width, action_tokens)
            y = functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            y = y.transpose(1, 2).reshape_as(x)
            return self.proj(y)

    class _MiniBlock(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(24)
            self.attn = _MiniAttention()
            self.drop_path = nn.Identity()
            self.norm2 = nn.LayerNorm(24)
            self.mlp = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))

        def forward(self, x: Any, **kwargs: Any) -> Any:
            x = x + self.attn(self.norm1(x), **kwargs)
            return x + self.mlp(self.norm2(x))

    class _MiniPredictor(nn.Module):  # type: ignore[name-defined,misc]
        grid_height = 1
        grid_width = 2
        use_extrinsics = False

        def __init__(self) -> None:
            super().__init__()
            self.predictor_embed = nn.Linear(6, 24)
            self.action_encoder = nn.Linear(3, 24)
            self.state_encoder = nn.Linear(3, 24)
            self.predictor_blocks = nn.ModuleList([_MiniBlock(), _MiniBlock()])
            self.predictor_norm = nn.LayerNorm(24)
            self.predictor_proj = nn.Linear(24, 6)
            block = 2 + self.grid_height * self.grid_width
            frame_ids = torch.arange(8 * block) // block
            self.attn_mask = frame_ids[:, None] >= frame_ids[None, :]

        def forward(self, visual: Any, actions: Any, states: Any) -> Any:
            batch, rows, _dim = visual.shape
            frames = rows // 2
            x = self.predictor_embed(visual).view(batch, frames, 2, 24)
            a = self.action_encoder(actions).unsqueeze(2)
            s = self.state_encoder(states).unsqueeze(2)
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)
            mask = self.attn_mask[: x.shape[1], : x.shape[1]]
            for block in self.predictor_blocks:
                x = block(
                    x,
                    attn_mask=mask,
                    T=frames,
                    H=1,
                    W=2,
                    action_tokens=2,
                )
            x = x.view(batch, frames, 4, 24)[:, :, 2:].flatten(1, 2)
            return self.predictor_proj(self.predictor_norm(x))

    return _MiniPredictor().eval()


def test_ac_predictor_adapter_cached_growth_matches_full_forward() -> None:
    """The real adapter's durable prefix keeps historical actions at zero.

    A miniature two-layer AC predictor exercises the same augmented token
    layout, axial RoPE, block-causal attention and residual stack.  Multiple
    cached steps must match the public full-forward contract, not merely the
    first step where accidentally persisting the live action is invisible.
    """
    torch = pytest.importorskip("torch")

    torch.manual_seed(4)
    adapter = _AcPredictorAdapter(_build_mini_ac_predictor(), tokens_per_frame=2, action_dim=3)
    for initial_rows in (2, 4):  # empty prefix and one-frame prefix bootstrap
        context = torch.randn(initial_rows, 6)
        cache = adapter.init_cache(context)
        for _ in range(3):
            action = torch.randn(3)
            expected = adapter(context, action)
            actual, staged = adapter.step_cached(cache, action)
            assert torch.allclose(actual, expected, atol=2e-6)
            cache = adapter.append_frame(staged, actual)
            context = torch.cat([context, actual], dim=0)


def test_ac_predictor_adapter_batched_step_cached_matches_per_candidate_loop() -> None:
    """ADR-0009's deferred CEM-batched reuse dimension, at the adapter level.

    S candidates advancing together through one shared, expanding-not-
    copying cache must match running the exact same single-candidate
    ``step_cached``/``append_frame`` path S times independently, one action
    vector at a time. This is the thing that would silently break if the
    batch expand in ``_advance_pending`` ever mixed candidates' K/V instead
    of giving each its own after divergence.
    """
    torch = pytest.importorskip("torch")

    torch.manual_seed(7)
    adapter = _AcPredictorAdapter(_build_mini_ac_predictor(), tokens_per_frame=2, action_dim=3)
    context = torch.randn(4, 6)  # 2-frame prefix + 1 pending frame
    s, horizon = 3, 3
    actions = torch.randn(s, horizon, 3)

    batched_cache = adapter.init_cache(context)
    batched_terminal = None
    for t in range(horizon):
        predicted, batched_cache = adapter.step_cached(batched_cache, actions[:, t])
        batched_terminal = predicted
        batched_cache = adapter.append_frame(batched_cache, predicted)
    assert batched_terminal is not None
    assert tuple(batched_terminal.shape) == (s, 2, 6)

    for i in range(s):
        cache = adapter.init_cache(context)
        terminal = None
        for t in range(horizon):
            predicted, cache = adapter.step_cached(cache, actions[i, t])
            terminal = predicted
            cache = adapter.append_frame(cache, predicted)
        assert terminal is not None
        assert torch.allclose(batched_terminal[i], terminal, atol=1e-5)


def test_step_appends_patch_token_frame_block() -> None:
    torch = pytest.importorskip("torch")
    # The real path emits P>1 patch tokens per frame; step must append the block
    # and cap the window by frames (not rows).
    p, d = 3, 4
    ctx0 = torch.zeros(2 * p, d)  # 2 frames

    class _BlockPredictor:
        def __call__(self, context: Any, action: Any) -> Any:
            return torch.ones(p, d) * action.sum()

    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=d, context_frames=2),
        encoder=_FakeEncoder(ctx0),
        predictor=_BlockPredictor(),
    )
    engine._tokens_per_frame = p  # real path sets this from the encoder config
    state = engine.reset(ConditioningInput(), RolloutParams())
    assert tuple(state.context.shape) == (2 * p, d)  # 2 frames kept
    nxt, _ = engine.step(state, Action(values=[1.0, 1.0, 1.0, 1.0]))
    assert tuple(nxt.context.shape) == (2 * p, d)  # appended P, capped to 2 frames


# --- seam loop contract (needs torch for the latent tensors) ---


def test_stub_step_streams_in_order() -> None:
    pytest.importorskip("torch")
    engine = _StubInteractive()
    state = engine.reset(ConditioningInput(), RolloutParams())
    seen: list[int] = []
    for _ in range(3):
        state, step = engine.step(state, Action(values=[0.0]))
        seen.append(step.step_index)
    assert seen == [1, 2, 3]
    assert state.step_index == 3


def test_reset_and_step_advance_context() -> None:
    torch = pytest.importorskip("torch")
    ctx0 = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    engine = _toy_engine(ctx0, context_frames=4)
    assert engine.info().ready is True

    state = engine.reset(ConditioningInput(), RolloutParams())
    assert state.step_index == 0
    assert tuple(state.context.shape) == (2, 4)

    nxt, step = engine.step(state, Action(values=[1.0, 1.0, 1.0, 1.0]))
    assert step.step_index == 1
    assert tuple(nxt.context.shape) == (3, 4)
    # Toy dynamics: new last frame == old last frame + action.
    assert torch.allclose(nxt.context[-1], state.context[-1] + torch.ones(4))


def test_context_window_is_capped() -> None:
    torch = pytest.importorskip("torch")
    ctx0 = torch.zeros(1, 3)
    engine = _toy_engine(ctx0, context_frames=2)
    state = engine.reset(ConditioningInput(), RolloutParams())
    for _ in range(5):
        state, _ = engine.step(state, Action(values=[0.0, 0.0, 0.0]))
    assert tuple(state.context.shape) == (2, 3)  # capped at context_frames


class _BatchedFakePredictor:
    """Linear toy dynamics that also accepts a candidate batch.

    Single: ``(N, D) + (A,) -> (D,)``; batched: ``(S, N, D) + (S, A) -> (S, 1, D)``
    (one "patch token" per frame, matching ``tokens_per_frame == 1``).
    """

    supports_batch = True

    def __call__(self, context: Any, action: Any) -> Any:
        if context.ndim == 3:
            return (context[:, -1] + action).unsqueeze(1)
        return context[-1] + action


def test_batched_candidate_energies_match_loop() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    ctx0 = torch.randn(2, 4)
    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=4),
        encoder=_FakeEncoder(ctx0),
        predictor=_BatchedFakePredictor(),
    )
    state = engine.reset(ConditioningInput(), RolloutParams())
    goal = torch.randn(4)
    seqs = torch.randn(5, 3, 4)  # S=5 candidates, H=3, A=4

    batched = engine._rollout_energy_batched(state, seqs, goal)
    looped = torch.stack([engine._rollout_energy(state, seqs[i], goal) for i in range(5)])
    assert tuple(batched.shape) == (5,)
    assert torch.allclose(batched, looped, atol=1e-5)


def test_plan_dispatches_to_batched_path() -> None:
    torch = pytest.importorskip("torch")

    class _CountingPredictor(_BatchedFakePredictor):
        calls: ClassVar[list[int]] = []

        def __call__(self, context: Any, action: Any) -> Any:
            self.calls.append(int(context.shape[0]) if context.ndim == 3 else 1)
            return super().__call__(context, action)

    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=4, plan_samples=6, plan_elites=2, plan_iters=1),
        encoder=_FakeEncoder(torch.zeros(2, 4)),
        predictor=_CountingPredictor(),
    )
    state = engine.reset(ConditioningInput(), RolloutParams())
    engine.plan(state, torch.zeros(4), horizon=2)
    # One batched forward of all 6 candidates per rollout timestep (H=2),
    # not 6 x 2 single forwards.
    assert _CountingPredictor.calls == [6, 6]


def test_warm_start_shifts_previous_plan() -> None:
    torch = pytest.importorskip("torch")
    engine = _toy_engine(torch.zeros(2, 4), context_frames=8)
    state = engine.reset(ConditioningInput(), RolloutParams())
    goal = torch.zeros(4)

    prev = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    engine._plan_mean[state.session_id] = prev
    mean = engine._warm_start_mean(state, goal, horizon=3)
    # Receding horizon: drop the executed first action, repeat the last row.
    assert torch.equal(mean, torch.cat([prev[1:], prev[-1:]], dim=0))
    # Horizon mismatch and unknown sessions fall back to the zero mean.
    assert torch.all(engine._warm_start_mean(state, goal, horizon=5) == 0)

    torch.manual_seed(0)
    engine.plan(state, goal, horizon=3)
    assert state.session_id in engine._plan_mean  # solution recorded for reuse

    engine._config.plan_warm_start = False
    assert torch.all(engine._warm_start_mean(state, goal, horizon=3) == 0)


def test_sdpa_dtype_harmonizer_casts_qk_to_v() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F  # noqa: N812

    from repercep.models.vjepa2_ac import _sdpa_dtype_harmonizer

    q = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    k = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    v = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    with _sdpa_dtype_harmonizer():
        out = F.scaled_dot_product_attention(q, k, v)
    assert out.dtype == torch.bfloat16
    # The patch is scoped: outside the guard the original op (and its dtype
    # strictness) is restored.
    with pytest.raises(Exception, match=r"dtype|scalar type"):
        F.scaled_dot_product_attention(q, k, v)


def test_plan_reduces_energy_toward_goal() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    engine = _toy_engine(torch.zeros(2, 4), context_frames=8)
    state = engine.reset(ConditioningInput(), RolloutParams())
    goal = state.context[-1] + torch.tensor([2.0, 0.0, -1.0, 0.5])

    e_zero = engine._rollout_energy(state, torch.zeros(3, 4), goal)
    sequence = engine._plan_sequence(state, goal, horizon=3)
    e_planned = engine._rollout_energy(state, sequence, goal)
    # CEM minimizes the terminal latent energy → planned beats the zero action.
    assert float(e_planned) < float(e_zero)

    action = engine.plan(state, goal, horizon=3)
    assert len(action.values) == 4
    assert action.space == "ee_delta"


# --- KV/latent reuse (ADR-0009): engine-side seam, step() path ---


class _KVCacheFakePredictor:
    """Toy dynamics for the KV-cache seam: next frame = sum(live frames) + action.

    Implements both the full-context call (the parity baseline) and the
    incremental ``_CachedPredictor`` methods. Sensitive to exactly which
    frames are "live" — dropping, duplicating, or staling a frame changes the
    sum, so a bookkeeping bug (bad eviction, cache leaking across a branch)
    produces a numerically WRONG result, not merely a slower one.

    ``step_cached`` predicts only (does not mutate ``cache``); the engine's
    layer-normed block comes back via ``append_frame`` — mirrors the real
    contract even though this toy predictor has no norm of its own.
    """

    supports_kv_cache = True

    def __init__(self) -> None:
        self.init_cache_calls = 0
        self.step_cached_calls = 0
        self.append_frame_calls = 0
        self.evict_calls = 0

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return context.sum(dim=0) + action

    def init_cache(self, context: torch.Tensor) -> list[torch.Tensor]:
        self.init_cache_calls += 1
        return list(context.unbind(0))

    def step_cached(
        self, cache: list[torch.Tensor], action: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        import torch

        self.step_cached_calls += 1
        total = torch.stack(cache, dim=0).sum(dim=0) if cache else torch.zeros_like(action)
        next_frame = total + action
        return next_frame, cache

    def append_frame(
        self, cache: list[torch.Tensor], normed_block: torch.Tensor
    ) -> list[torch.Tensor]:
        self.append_frame_calls += 1
        return [*cache, *normed_block.unbind(0)]

    def evict(self, cache: list[torch.Tensor], sink_frames: int = 0) -> list[torch.Tensor]:
        self.evict_calls += 1
        return cache[:sink_frames] + cache[sink_frames + 1 :]


def test_kv_cache_matches_full_recompute_with_slide_eviction() -> None:
    """Cached step() must match full-recompute step() exactly, including
    after the window fills and starts evicting, under the opt-in "slide"
    policy — the case a naive append-only cache would get wrong (ADR-0009's
    RoPE-shift discussion); this fake's toy dynamics make a wrong eviction
    numerically visible. (The real predictor's evict is NOT exact — ADR-0009
    Finding 2 — this only proves the engine calls append/evict correctly
    given a predictor whose evict happens to be, like this fake's trivial
    list-drop.)
    """
    torch = pytest.importorskip("torch")
    ctx0 = torch.randn(2, 4)

    cached_pred = _KVCacheFakePredictor()
    cached_engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(
            action_dim=4, context_frames=3, use_kv_cache=True, kv_evict_policy="slide"
        ),
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=cached_pred,
    )
    full_pred = _KVCacheFakePredictor()
    full_engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=3, use_kv_cache=False),
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=full_pred,
    )
    cached_state = cached_engine.reset(ConditioningInput(), RolloutParams())
    full_state = full_engine.reset(ConditioningInput(), RolloutParams())

    torch.manual_seed(1)
    for _ in range(6):  # crosses the context_frames=3 cap (window fills at step 1)
        action = Action(values=torch.randn(4).tolist())
        cached_state, _ = cached_engine.step(cached_state, action)
        full_state, _ = full_engine.step(full_state, action)
        assert torch.allclose(cached_state.context, full_state.context)

    # Real incremental work happened, not a silent full-recompute every call.
    assert cached_pred.step_cached_calls == 6
    assert cached_pred.init_cache_calls == 1  # bootstrapped once, then reused
    assert cached_pred.append_frame_calls == 6  # every predicted frame durably cached
    assert cached_pred.evict_calls == 5  # every step once the window is full


def test_kv_cache_reinit_policy_never_evicts_but_degrades_gracefully() -> None:
    """The default "reinit" policy must still match full-recompute exactly
    (dropping a cache and rebuilding it via init_cache can never be wrong),
    but — honestly — buys no incremental savings once a strictly-capped
    window saturates: every step past the growth phase uses the ordinary
    full forward, same cost as use_kv_cache=False. This is the graceful-
    degradation behavior ADR-0009 §"Revisit if" settles on in place of the
    unsound "slide" default; this test pins the exact call-count evidence
    for it so a future change can't silently regress it back to unsound.
    """
    torch = pytest.importorskip("torch")
    ctx0 = torch.randn(2, 4)

    cached_pred = _KVCacheFakePredictor()
    cached_engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=3, use_kv_cache=True),  # reinit is default
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=cached_pred,
    )
    full_pred = _KVCacheFakePredictor()
    full_engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=3, use_kv_cache=False),
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=full_pred,
    )
    cached_state = cached_engine.reset(ConditioningInput(), RolloutParams())
    full_state = full_engine.reset(ConditioningInput(), RolloutParams())

    torch.manual_seed(1)
    for _ in range(6):
        action = Action(values=torch.randn(4).tolist())
        cached_state, _ = cached_engine.step(cached_state, action)
        full_state, _ = full_engine.step(full_state, action)
        assert torch.allclose(cached_state.context, full_state.context)

    assert cached_engine._config.kv_evict_policy == "reinit"
    assert cached_pred.evict_calls == 0  # never calls the unsound operation
    # step 1: window not yet full (2 < 3) -> cheap growth, cache built + kept.
    # step 2: window full on entry, but the cache built in step 1 is still
    # valid for this one prediction -> reused, then dropped afterward (no
    # append -- there's nowhere sound to put the appended frame).
    # steps 3-6: no cache survives from the previous step -> direct ordinary
    # full forwards (the engine avoids paying init_cache() just to discard it).
    # Net: 1 bootstrap, one cheap reuse (step 2), then graceful full-forward
    # fallback at exactly the same cost as use_kv_cache=False.
    assert cached_pred.init_cache_calls == 1
    assert cached_pred.step_cached_calls == 2
    assert cached_pred.append_frame_calls == 1  # only the one cheap growth step


def test_kv_cache_does_not_leak_across_branches() -> None:
    """A rollout that restarts from the SAME state under the SAME session —
    exactly what CEM's sequential ``_rollout_energy`` does per candidate —
    must not let a later candidate silently resume an earlier candidate's
    cache. Without the context-sync check this fake's sum-of-frames output
    would come out wrong for the second candidate; compared against a
    caching-disabled reference to prove it doesn't.
    """
    torch = pytest.importorskip("torch")
    ctx0 = torch.randn(2, 4)
    goal = torch.randn(4)
    seq_a = torch.randn(3, 4)
    seq_b = torch.randn(3, 4)

    cached = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=8, use_kv_cache=True),
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=_KVCacheFakePredictor(),
    )
    state = cached.reset(ConditioningInput(), RolloutParams())
    energy_a = cached._rollout_energy(state, seq_a, goal)
    energy_b = cached._rollout_energy(state, seq_b, goal)  # same `state` — a branch

    reference = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=8, use_kv_cache=False),
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=_KVCacheFakePredictor(),
    )
    ref_state = reference.reset(ConditioningInput(), RolloutParams())
    ref_a = reference._rollout_energy(ref_state, seq_a, goal)
    ref_b = reference._rollout_energy(ref_state, seq_b, goal)

    assert torch.allclose(energy_a, ref_a)
    assert torch.allclose(energy_b, ref_b)


def test_release_drops_session_kv_cache_and_warm_start_mean() -> None:
    """``release()`` frees a session's server-side state (the leak fix).

    Without this, a churn of short-lived WebSocket sessions leaks one KV-
    cache entry (real GPU tensors on the real predictor) per session
    forever — nothing previously removed an entry on session *end*, only
    mid-session on a saturated "reinit" cache (see ``step()``).
    """
    torch = pytest.importorskip("torch")
    ctx0 = torch.randn(2, 4)
    goal = torch.randn(4)
    seq = torch.randn(2, 4)

    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=8, use_kv_cache=True, plan_warm_start=True),
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=_KVCacheFakePredictor(),
    )
    state = engine.reset(ConditioningInput(), RolloutParams())
    engine._plan_sequence(state, goal, horizon=2)  # seeds _plan_mean
    engine._rollout_energy(state, seq, goal)  # seeds _kv_cache via step()
    assert state.session_id in engine._plan_mean
    assert state.session_id in engine._kv_cache

    engine.release(state)

    assert state.session_id not in engine._plan_mean
    assert state.session_id not in engine._kv_cache

    # Releasing an already-released (or never-seen) session is a no-op, not
    # an error — the serving layer's finally-block call site can't always
    # know whether reset() completed before a disconnect.
    engine.release(state)


def test_kv_cache_disabled_when_predictor_lacks_support() -> None:
    """A predictor without ``supports_kv_cache`` keeps using the existing
    full-recompute path — no behavior change for today's real predictor
    until the GPU-verified cache wrapper lands (ADR-0009)."""
    torch = pytest.importorskip("torch")
    ctx0 = torch.zeros(2, 4)
    engine = _toy_engine(ctx0, context_frames=4)  # _FakePredictor: no cache support
    assert getattr(engine._predictor, "supports_kv_cache", False) is False
    state = engine.reset(ConditioningInput(), RolloutParams())
    nxt, _ = engine.step(state, Action(values=[1.0, 1.0, 1.0, 1.0]))
    assert engine._kv_cache == {}  # never touched
    assert torch.allclose(nxt.context[-1], state.context[-1] + torch.ones(4))


# --- KV/latent reuse (ADR-0009): engine-side seam, CEM-batched plan() path ---


class _BatchedKVCacheFakePredictor:
    """Toy dynamics for the batched-KV-cache seam: next frame = (sum of all
    live frames so far) + action, generalized to a CEM candidate batch (S).

    The cache IS the running sum (sum is associative, so appending a frame
    just adds it in) — starts batch-1 (one shared prefix) and naturally
    broadcasts to batch-S the first time it's added to an (S, D) action/
    normed-block, exactly mirroring the real adapter's expand-on-first-
    divergence behavior, without needing explicit ``.expand()`` calls in this
    toy. Same "wrong sum = wrong answer, not just slower" sensitivity as
    ``_KVCacheFakePredictor``, generalized to the batch dimension.
    """

    supports_kv_cache = True
    supports_batch = True

    def __init__(self) -> None:
        self.init_cache_calls = 0

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if context.ndim == 3:
            return (context.sum(dim=1) + action).unsqueeze(1)
        return context.sum(dim=0) + action

    def init_cache(self, context: torch.Tensor) -> torch.Tensor:
        self.init_cache_calls += 1
        return context.sum(dim=0)  # (D,): batch-1 shared-prefix running sum

    def step_cached(
        self, cache: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cache + action, cache  # predict only; cache unchanged until append

    def append_frame(self, cache: torch.Tensor, normed_block: torch.Tensor) -> torch.Tensor:
        return cache + normed_block  # running sum grows; broadcasts (D,) -> (S, D)


def test_rollout_energy_batched_cached_matches_uncached_batched_path() -> None:
    """The KV-cached CEM-batched rollout must match the plain batched path
    (which itself is proven against the per-candidate loop by
    ``test_batched_candidate_energies_match_loop``) — same answer, cheaper.
    Growing-window precondition holds here (frames + horizon <= cap).
    """
    torch = pytest.importorskip("torch")
    torch.manual_seed(2)
    ctx0 = torch.randn(2, 4)
    goal = torch.randn(4)
    seqs = torch.randn(5, 3, 4)  # S=5, H=3, A=4

    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(
            action_dim=4, context_frames=8, use_kv_cache=True, plan_batched_kv=True
        ),
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=_BatchedKVCacheFakePredictor(),
    )
    state = engine.reset(ConditioningInput(), RolloutParams())

    cached = engine._rollout_energy_batched_cached(state, seqs, goal)
    uncached = engine._rollout_energy_batched(state, seqs, goal)
    assert cached is not None
    assert torch.allclose(cached, uncached, atol=1e-5)

    # And the public dispatch (_candidate_energies -> plan()) actually
    # reaches the cached path when configured on, not just when called
    # directly.
    predictor = cast("_BatchedKVCacheFakePredictor", engine._predictor)
    calls_before = predictor.init_cache_calls
    dispatched = engine._candidate_energies(state, seqs, goal)
    assert torch.allclose(dispatched, uncached, atol=1e-5)
    assert predictor.init_cache_calls == calls_before + 1


def test_rollout_energy_batched_cached_falls_back_past_growing_window() -> None:
    """Once frames_now + horizon would exceed the cap, the cached path
    returns ``None`` (never a wrong answer) instead of using a cache it
    isn't sound for — ``_candidate_energies`` then falls back to the plain
    batched path, which must still land on the same, correct answer.
    """
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    ctx0 = torch.randn(3, 4)  # 3 frames already -- context_frames=4, horizon=3 overflows
    goal = torch.randn(4)
    seqs = torch.randn(4, 3, 4)  # S=4, H=3, A=4

    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(
            action_dim=4, context_frames=4, use_kv_cache=True, plan_batched_kv=True
        ),
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=_BatchedKVCacheFakePredictor(),
    )
    state = engine.reset(ConditioningInput(), RolloutParams())

    assert engine._rollout_energy_batched_cached(state, seqs, goal) is None
    dispatched = engine._candidate_energies(state, seqs, goal)
    reference = engine._rollout_energy_batched(state, seqs, goal)
    assert torch.allclose(dispatched, reference)


def test_rollout_energy_batched_cached_disabled_by_default() -> None:
    """``plan_batched_kv`` defaults to ``False`` — GPU-verify is pending
    (CPU-parity-tested only so far), so opting in must be explicit."""
    torch = pytest.importorskip("torch")
    ctx0 = torch.randn(2, 4)
    goal = torch.randn(4)
    seqs = torch.randn(3, 2, 4)

    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=4, context_frames=8, use_kv_cache=True),  # plan_batched_kv unset
        encoder=_FakeEncoder(ctx0.clone()),
        predictor=_BatchedKVCacheFakePredictor(),
    )
    assert engine._config.plan_batched_kv is False
    state = engine.reset(ConditioningInput(), RolloutParams())
    predictor = cast("_BatchedKVCacheFakePredictor", engine._predictor)

    engine._candidate_energies(state, seqs, goal)
    assert predictor.init_cache_calls == 0  # cached path never even attempted
