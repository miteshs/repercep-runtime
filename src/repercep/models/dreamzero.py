"""DreamZero engine — the third model on the interactive seam.

DreamZero (NVIDIA GEAR, 2026) is an autoregressive **world-action model**: a
``CausalWanModel`` — Wan2.1-I2V-14B-480P DiT extended with action/state
registers — over Wan2.1 VAE latents, trained with flow matching, generating
**block-by-block** with a per-layer KV cache. Per block it jointly denoises
video-latent noise and action-register noise; the action-register slice
decodes into an executed action chunk. Same policy regime as LingBot-VA (the
model proposes its own actions; no external CEM), but a different
backbone/cache shape and attached to the GR00T 2 ecosystem. See
``docs/DREAMZERO_PORT_PLAN.md`` for the full scoping (interfaces verified
against ``dreamzero0/dreamzero`` ``socket_test_optimized_AR.py`` +
``groot/vla/model/dreamzero/``, and §2b for the GPU-verified real forward
pass, 2026-07-13).

This module wraps it as an
:class:`~repercep.runtime.interactive.InteractiveWorldModel`. What is
**implemented and tested** here is the model-agnostic chunk-loop layer:
session bookkeeping, the recondition-then-predict :meth:`DreamZeroEngine.step`,
and the policy-mode :meth:`DreamZeroEngine.plan`. They run against an
injected ``pipeline`` (see :class:`_DreamZeroPipeline`), so the loop is
exercised on CPU without weights — the ``vjepa2_ac.py`` / ``lingbot_va.py``
pattern. The model-specific pipeline
(:mod:`repercep.models.dreamzero_pipeline`, Phase 1) ``sys.path``-inserts the
research repo rather than vendoring or ``pip install``-ing it (same
discipline as ``lingbot_va_pipeline.py``'s ``import wan_va``); its denoise
loop is GPU-verified working, but the pipeline module itself hasn't yet been
exercised through this engine's chunk loop end-to-end (only via a standalone
smoke-test script) — see the port plan §2b for exactly what's confirmed vs.
still open.

Chunk-loop geometry note (the DreamZero-DROID checkpoint's actual numbers —
confirmed against its real ``config.json``, not the research repo's generic
demo/socket-server comments, which describe a different geometry — see the
port plan §1's correction): one block is **two** latent frames
(``num_frame_per_block=2``); the executed unit per :meth:`step`/:meth:`plan`
is one **action** chunk of ``num_action_per_block=24`` actions,
``action_dim=32`` padded (DROID's real wire width is 8 — see
``DreamZeroConfig.used_action_dim``). The KV cache carries a full
negative-prompt copy for CFG plus a 512-token cross-attn cache, and resets
when the attention window (``attn_window_frames=21`` frames) fills or the
task/language changes — bookkeeping the pipeline owns, not this layer.

State note (same caveat as LingBot-VA, **plus a further wrinkle verified on
GPU**): the native serving state is a per-session mutable KV cache inside
the transformer, so the seam's "previous state is not mutated" branching
guarantee does not hold in v0 — the engine keeps one live branch per
session. Worse than LingBot-VA's case: ``WANPolicyHead`` has *no* named
multi-session cache API at all (unlike ``wan_va``'s ``cache_name`` kwarg) —
its cache/clock/language state are plain mutable instance attributes on one
``nn.Module``, so ``dreamzero_pipeline.py`` implements its own session-swap
(save/restore those attributes around every call) rather than delegating to
a native per-session keying mechanism. A base shared with ``lingbot_va``
("chunked VA over a Wan-family causal DiT with named KV cache") is a
Phase-0 design question per the port plan, not a commitment made here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from repercep.runtime.engine import EngineInfo
from repercep.runtime.types import Action, LatentStep, WorldState

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend
    from repercep.runtime.types import ConditioningInput, RolloutParams

#: The inference-ready DROID checkpoint (HF, Apache-2.0). ``GEAR-Dreams/DreamZero-AgiBot``
#: is the post-training bundle (new embodiments), not the serving default.
DEFAULT_REPO = "GEAR-Dreams/DreamZero-DROID"


class _DreamZeroPipeline(Protocol):
    """The model-specific half of the port, as an injectable session pipeline.

    Mirrors ``WANPolicyHead.lazy_joint_video_action`` in
    ``groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`` —
    the real per-block entry point, two dispatch layers below the eval-harness
    wrapper (``ARDroidRoboarenaPolicy``, never ported — same discipline as
    LingBot-VA's unported ``VA_Server``). ``reset`` builds the per-layer KV
    cache (+ CFG negative-prompt copy + 512-token cross-attn caches) and
    encodes the text prompt, ``encode_observation`` is the first-frame
    CLIP/VAE encode. The reference bundles what this Protocol splits into
    ``recondition`` + ``infer_chunk`` into a *single* call
    (``lazy_joint_video_action(..., latent_video=...)``: push the caller's new
    frames into the cache, then run the joint video+action denoise loop for
    the next block) — there is no separate "just push" entrypoint upstream.
    ``recondition`` here should stash the executed actions / obs latent on the
    session; ``infer_chunk`` does the real work, passing the stashed value
    through as ``latent_video`` (port plan §2). Keeping this a Protocol
    isolates the chunk-loop logic from the port, so the loop is testable on
    CPU with fakes regardless of how a Phase-1 implementation splits the work.
    """

    def reset(self, session_id: str, prompt: str | None) -> None: ...

    def encode_observation(
        self, session_id: str, conditioning: ConditioningInput
    ) -> torch.Tensor: ...

    def infer_chunk(
        self, session_id: str, current_start_frame: int, init_latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one block → ``(latents (num_frame_per_block, D), actions (K, action_dim))``."""
        ...

    def recondition(
        self,
        session_id: str,
        actions: torch.Tensor,
        obs_latent: torch.Tensor | None,
        current_start_frame: int,
    ) -> int:
        """Push executed actions (+ real obs when available) into the cache.

        Returns the number of latent frames consumed (advances
        ``current_start_frame``). ``obs_latent=None`` is imagination mode:
        the model's own predicted latents (already cached by
        :meth:`infer_chunk`) stand in for reality.
        """
        ...

    def close(self, session_id: str) -> None:
        """Drop the session's named KV + cross-attn caches. The other half of ``reset``."""
        ...


@dataclass(slots=True)
class DreamZeroConfig:
    """Load-time + rollout configuration for :class:`DreamZeroEngine`.

    Defaults are read from the **actual released checkpoint's `config.json`**
    (``GEAR-Dreams/DreamZero-DROID``, fetched 2026-07-13 on an H100 pod — not
    the research repo's generic demo config, which uses different geometry:
    the demo/socket-server comments say ``num_frame_per_block=1``,
    ``num_action_per_block=32``, but the shipped DROID checkpoint's DiT was
    trained with ``num_frame_per_block=2``, ``num_action_per_block=24``,
    ``max_chunk_size=4`` — these are structural (baked into RoPE/registers),
    not a runtime choice, so they must match the checkpoint being served, not
    the docs). 3-camera 880-token frames, a 21-frame (~18.5k token) attention
    window, 16-step joint flow matching, CFG 5.0, sigma shift 5.0, bf16, SDPA
    attention (the ROCm-portable path — flash-attn is optional and
    try/except-guarded in the reference code, no stub needed unlike
    LingBot-VA).
    """

    repo: str = DEFAULT_REPO
    device_index: int = 0
    dtype: str = "bfloat16"
    # The task instruction — DreamZero's goal conditioning is textual (umT5).
    prompt: str | None = None
    # Chunked generation geometry -- DreamZero-DROID checkpoint values (see
    # class docstring), not the research repo's generic demo/socket-server
    # defaults.
    num_frame_per_block: int = 2
    frame_seqlen: int = 880  # tokens/frame across 2 exterior + 1 wrist camera
    num_action_per_block: int = 24
    num_state_per_block: int = 1
    # The model's padded action-register width (config.json `action_dim` /
    # `max_action_dim` -- DreamZero-DROID is trained jointly across multiple
    # embodiments' action spaces zero-padded into one 32-wide channel).
    action_dim: int = 32
    # DROID's *used* wire width, confirmed against the checkpoint's
    # experiment_cfg/conf.yaml (2026-07-13): action_concat_order =
    # [action.joint_position (7), action.gripper_position (1)] = 8, occupying
    # indices [0:8] of the padded action_dim=32 tensor (zero-padded at the
    # end, per DreamTransform's `np.pad(..., (0, max_action_dim - n))`) --
    # NOT the cartesian-delta 7-DoF this field previously guessed. Denorm is
    # q01/q99 quantile per channel from experiment_cfg/metadata.json's
    # `statistics.action.{joint_position,gripper_position}` -- same formula
    # as LingBotVAConfig's used_action_dim / `_denormalize_actions`, verified
    # against groot/vla/data/transform/state_action.py's `StateActionTransform`
    # (mode="q99": `(x+1)/2*(q99-q01)+q01`), which we do NOT depend on --
    # reimplemented directly, LingBot-VA style.
    used_action_dim: int | None = 8
    # The state encoder's padded width -- a DISTINCT number from action_dim,
    # confirmed by a GPU shape-mismatch crash 2026-07-13 (expected [1,8], got
    # [1,64] against the embodiment-keyed state-encoder weight): DROID's real
    # state (state.joint_position(7)+state.gripper_position(1)=8, same
    # channels as the action) is zero-padded to 64, not 32. No used_state_dim
    # field yet -- Phase-1 pipeline TODO, mirrors used_action_dim once
    # reset()/encode_observation() actually construct this tensor.
    max_state_dim: int = 64
    # attn window in frames (default: local_attn_size=-1 sentinel in the
    # reference DiT config -> max_attention_size = attn_window_frames * frame_seqlen;
    # a non-default local_attn_size overrides this directly, in frames).
    attn_window_frames: int = 21
    # Flow-matching denoise budget per block: one joint loop over both the
    # video and action noise (two FlowUniPCMultistepScheduler instances, same
    # step index), not two separate loops. Confirmed by reading
    # WANPolicyHead.lazy_joint_video_action (port plan §2): 16 is the only
    # value actually used — WANPolicyHeadConfig.num_inference_timesteps is
    # read into an attribute but never referenced by that method (dead for
    # this path).
    num_inference_steps: int = 16
    cfg_scale: float = 5.0
    sigma_shift: float = 5.0
    # Their headline latency lever: a dynamic DiT-call skip schedule (cosine
    # similarity of the last two action-noise predictions >0.95/0.93 skips
    # the next 4/2 calls, reusing the last prediction verbatim) — off by
    # default, quality impact is ours to measure, not the pipeline's default
    # stance. Same construction-time-only caveat as num_dit_steps above (same
    # WANPolicyHead.__init__ block, read via DYNAMIC_CACHE_SCHEDULE) --
    # requires a fresh pipeline to change, GPU-verified 2026-07-14.
    enable_dit_cache: bool = False
    attn_mode: str = "sdpa"
    # ip=2 only splits CFG (rank 0 conditional / rank 1 unconditional,
    # exchanged via P2P) — a convenience, not a sharding requirement. ip=1
    # runs both contexts sequentially in one process; batching them into one
    # forward is the obvious single-GPU lever (port plan §2.5), a pipeline
    # concern this config doesn't encode.
    ip_size: int = 1
    # --- Phase-2 latency levers (docs/DREAMZERO_PORT_PLAN.md §4) ---
    # The reference's static DiT-call-skip cap (`NUM_DIT_STEPS` env, port plan
    # §2 point 5) defaults to 8 *unconditionally*, with no opt-in flag — so
    # their own "~3s/chunk H100" claim is already running an approximation.
    # GPU-verified 2026-07-14 (wan_flow_matching_action_tf.py's
    # WANPolicyHead.__init__, read directly): only {5, 6, 7, 8} have
    # hand-tuned partial masks -- any OTHER value, including this field's own
    # default of 16, falls through to the full 16-of-16 (all-True) mask. 16
    # is not "16 discrete skip levels"; it is simply the least surprising
    # sentinel for "full compute", picked because it equals num_inference_steps.
    # This env var is read ONCE at pipeline construction
    # (dreamzero_pipeline.build_pipeline), never per-call -- changing it on an
    # already-loaded engine's config does nothing; a fresh pipeline is
    # required (see build_pipeline's docstring for the full story, including
    # the silent-no-op this caused before it was caught on the first H100
    # run).
    num_dit_steps: int = 16
    # Batch cond+uncond into one forward instead of the reference's two
    # sequential calls per denoise step (port plan §2 point 4/§4 lever 1) —
    # the single-GPU differentiator vs. their 2-GPU ip=2 split, which only
    # splits the same two sequential calls across ranks. Needs pipeline
    # support (dreamzero_pipeline.py); the pipeline warns and falls back to
    # sequential CFG until the batching is GPU-verified against the research
    # clone's actual `_run_diffusion_steps`.
    cfg_batched: bool = False
    # torch.compile the transformer at pipeline-construction time (mirrors
    # LingBotVAConfig.compile_transformer). Phase 1 forced eager mode
    # (TORCHDYNAMO_DISABLE=1) after hitting FailOnRecompileLimitHit from
    # rank-changing history tensors (port plan §2b) — this flag is the
    # Phase-2 lever to retry that, isolated per engine load rather than a
    # blanket env var.
    compile: bool = False
    # KV-window override, in frames. None keeps the checkpoint's own default
    # (local_attn_size=-1 sentinel -> max_attention_size =
    # attn_window_frames * frame_seqlen, i.e. attn_window_frames above). A
    # smaller window trades context for KV memory (port plan §4 rung 5) — the
    # lever most directly connected to the resident-sessions/GPU headline.
    local_attn_size: int | None = None


@dataclass(slots=True)
class _Session:
    """Engine-held per-session state the wire-safe ``WorldState`` cannot carry."""

    current_start_frame: int = 0
    pending_actions: torch.Tensor | None = field(default=None)


class DreamZeroEngine:
    """DreamZero served on a Repercep backend (the interactive seam).

    The chunk loop (recondition → predict) and session bookkeeping are
    implemented model-agnostically against an injected
    :class:`_DreamZeroPipeline`; in production the pipeline would be built
    from the HF bundle on the first :meth:`load` (the Phase-1 port, not
    landed — see :meth:`load`), in tests it is a fake.
    ``WorldState.context`` carries the latest predicted latent block
    ``(num_frame_per_block, D)``; the KV cache lives in the pipeline keyed by
    ``session_id`` (see the module docstring's state note).
    """

    model_name = "dreamzero-droid"

    def __init__(
        self,
        backend: Backend,
        config: DreamZeroConfig | None = None,
        *,
        pipeline: _DreamZeroPipeline | None = None,
    ) -> None:
        self._backend = backend
        self._config = config if config is not None else DreamZeroConfig()
        # Injected for testing / advanced use; otherwise built by ``load()``
        # once Phase 1 lands.
        self._pipeline: _DreamZeroPipeline | None = pipeline
        self._sessions: dict[str, _Session] = {}

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name=self.model_name,
            backend=self._backend.name,
            device=f"{self._backend.name}:{self._config.device_index}",
            dtype=self._config.dtype,
            ready=self.is_loaded,
        )

    # --- loading (the model-specific port, Phase 1 — not landed) ---

    def load(self) -> None:
        """Build the real pipeline (GPU: needs ``groot`` importable + the checkpoint).

        The model-specific half lives in
        :mod:`repercep.models.dreamzero_pipeline` (Phase 1, GPU-verified on
        an H100 2026-07-13 — see ``docs/DREAMZERO_PORT_PLAN.md`` §2b): the
        research repo is ``sys.path``-inserted (``REPERCEP_DREAMZERO_SRC``
        env var, or already on ``sys.path``) rather than vendored or
        ``pip install``-ed — its pyproject hard-requires ``tensorrt``,
        ``ray``, ``mujoco``, ``deepspeed``, and a ``gear`` package that plain
        imports never touch (and ``tensorrt`` won't install on ROCm). Raises
        ``RuntimeError`` with the setup recipe when the research package
        isn't importable, same as :meth:`LingBotVAEngine.load`. Idempotent.
        """
        if self._pipeline is not None:
            return
        from repercep.models.dreamzero_pipeline import build_pipeline

        self._pipeline = build_pipeline(self._backend, self._config)

    # --- the interactive seam ---

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Open a session: build caches, encode the seed observation."""
        self._require_pipeline()
        assert self._pipeline is not None
        session_id = uuid.uuid4().hex
        self._pipeline.reset(session_id, self._config.prompt)
        init_latent = self._pipeline.encode_observation(session_id, conditioning)
        self._sessions[session_id] = _Session()
        return WorldState(context=init_latent, step_index=0, session_id=session_id)

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one block under the *executed* action chunk.

        Native order (the reference client loop): the previous prediction's
        cache entries are dropped and executed reality is pushed
        (``recondition``), then the next block is predicted. ``action.values``
        is a flat executed chunk — ``k x action_dim`` floats,
        ``space="dreamzero_chunk_droid"``. The model's *proposed* actions for
        the new block are parked on the session for a following
        :meth:`plan` call.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        session = self._session_for(state)
        actions = self._parse_action_chunk(action, like=state.context)
        consumed = self._pipeline.recondition(
            state.session_id, actions, None, session.current_start_frame
        )
        session.current_start_frame += consumed
        latents, proposed = self._pipeline.infer_chunk(
            state.session_id, session.current_start_frame, None
        )
        session.pending_actions = proposed
        new_state = WorldState(
            context=latents, step_index=state.step_index + 1, session_id=state.session_id
        )
        return new_state, LatentStep(step_index=new_state.step_index)

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Policy-mode planning: the model's own proposed next action chunk.

        DreamZero *generates* actions (the action-register denoise loop is
        the planner); the goal is the text prompt fixed at :meth:`reset`, so
        ``goal``/``horizon`` are accepted for seam compatibility and unused
        in v0. Returns the first action of the block proposed for the
        current state — predicting a fresh block if :meth:`step` has not
        already parked one.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        session = self._session_for(state)
        proposed = session.pending_actions
        if proposed is None:
            _, proposed = self._pipeline.infer_chunk(
                state.session_id, session.current_start_frame, state.context
            )
            session.pending_actions = proposed
        return Action(values=proposed[0].tolist(), space="dreamzero_chunk_droid")

    def release(self, state: WorldState) -> None:
        """Drop this session's server-side state (named KV cache + bookkeeping).

        Not part of :class:`InteractiveWorldModel` — the serving layer calls
        it duck-typed (``getattr(engine, "release", None)``) when a client
        session ends. Two layers hold state per session: this engine's own
        ``_sessions`` (frame clock, parked action proposal) and the
        pipeline's session-keyed KV/cross-attn caches (the actual GPU
        tensors) — both need dropping, or the cache leaks one entry per
        session forever under churn (the LingBot-VA lesson).
        """
        self._sessions.pop(state.session_id, None)
        if self._pipeline is not None:
            self._pipeline.close(state.session_id)

    # --- internals ---

    def _require_pipeline(self) -> None:
        if self._pipeline is None:
            self.load()

    def _session_for(self, state: WorldState) -> _Session:
        session = self._sessions.get(state.session_id)
        if session is None:
            raise KeyError(
                f"unknown session {state.session_id!r} — WorldState must come "
                "from this engine's reset()"
            )
        return session

    def _wire_action_dim(self) -> int:
        """The executed-action-chunk width (see ``DreamZeroConfig.used_action_dim``)."""
        return self._config.used_action_dim or self._config.action_dim

    def _parse_action_chunk(self, action: Action, *, like: torch.Tensor) -> torch.Tensor:
        """Validate + shape a flat executed chunk to ``(k, wire_action_dim)``."""
        import torch

        a_dim = self._wire_action_dim()
        if len(action.values) % a_dim != 0:
            raise ValueError(
                f"action chunk length {len(action.values)} is not a multiple "
                f"of action_dim={a_dim} (space={action.space!r})"
            )
        vec = torch.tensor(action.values, dtype=torch.float32, device=like.device)
        return vec.reshape(-1, a_dim)
