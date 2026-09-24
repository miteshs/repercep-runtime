"""VLA action-token engine — an autoregressive token-decoding VLA on the seam.

OpenVLA / π0-FAST / RT-2 class: a vision-language backbone (Llama / PaLI) that
decodes **action tokens** autoregressively with a KV cache. Serving one *is* LLM
inference — but control-camp: the output is a robot action, not chat, so it
lands on the :class:`~repercep.runtime.interactive.InteractiveWorldModel` seam
next to V-JEPA 2-AC and LingBot-VA rather than on a ``/v1/chat/completions``
surface (that lane is the co-located proxy, ``docs/LLM_PROXY.md``).

Why it belongs here and not on a paged-attention LLM stack: action horizons are
**short** — a chunk of ``action_chunk`` tokens per step, not thousands of
context tokens — so the growing-window KV + **CEM-candidate batching** substrate
already built for V-JEPA 2-AC (``models/vjepa2_ac.py``, ``plan_batched``) is the
right machine, and paged attention / big-sequence continuous batching are not
needed. That is the whole reason this is cheap and on-thesis where a general LLM
engine would be neither. See ``docs/VLA_PORT_PLAN.md``.

What is **implemented and tested** here is the model-agnostic layer: session
bookkeeping, the executed-action→advance→decode :meth:`VLAEngine.step`, and the
candidate-batched :meth:`VLAEngine.plan`. They run against an injected
``pipeline`` (see :class:`_VLAPipeline`), so the loop is exercised on CPU with a
fake — the ``lingbot_va.py`` pattern. The model-specific pipeline (processor,
action detokenizer, KV-cached backbone decode) is the Phase-1 GPU port:
:meth:`VLAEngine.load` raises ``NotImplementedError`` with the recipe until it
lands.

Regime note (the dual-planning story on one seam): V-JEPA 2-AC plans by
*searching* actions against a latent energy; LingBot-VA plans by *generating*
actions from a denoiser; a token VLA plans by *decoding* candidate action-token
sequences and scoring them. All three satisfy the same Protocol; ``plan``
differs in mechanism, not in contract.
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

#: The pretrained bundle. OpenVLA-7B is the lead token-VLA (Llama-2 backbone,
#: 256-bin per-dim action discretization, single-step decode). π0-FAST and
#: RT-2-style checkpoints implement the same seam with a different tokenizer.
DEFAULT_REPO = "openvla/openvla-7b"


class _VLAPipeline(Protocol):
    """The model-specific half of the port, as an injectable session pipeline.

    Keeping this a Protocol isolates the action-loop logic from the backbone,
    so the loop is testable on CPU with a fake. ``supports_batch`` advertises
    whether :meth:`decode_action_chunk` can score ``n_candidates`` in one
    forward (shared prefix KV) — the CEM-candidate-batching lever ``plan()``
    uses; a pipeline that can't sets it ``False`` and ``plan`` falls back to a
    per-candidate loop that is correct but slower.
    """

    supports_batch: bool

    def reset(self, session_id: str, prompt: str | None) -> None:
        """Open a session: build the KV cache and the instruction embedding."""
        ...

    def encode_observation(self, session_id: str, conditioning: ConditioningInput) -> torch.Tensor:
        """Encode the seed observation to the initial decode context ``(T, D)``."""
        ...

    def decode_action_chunk(
        self, session_id: str, context: torch.Tensor, n_candidates: int
    ) -> torch.Tensor:
        """Autoregressively decode ``n_candidates`` action-token chunks.

        Returns a detokenized ``(n_candidates, action_chunk, action_dim)``
        tensor of continuous actions. ``n_candidates == 1`` is the greedy step
        path; ``n_candidates > 1`` is the planner's candidate set, which shares
        the prompt+observation prefix KV across candidates (the batching lever).
        """
        ...

    def append_executed(
        self, session_id: str, executed: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Push the executed action chunk into the KV context; return the
        advanced context ``(T, D)`` the next decode attends over."""
        ...

    def score_candidates(
        self, session_id: str, candidates: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """Score each candidate against ``goal``; lower is better. Returns
        ``(n_candidates,)`` costs — the scalar :meth:`VLAEngine.plan` minimizes."""
        ...

    def close(self, session_id: str) -> None:
        """Drop the session's KV cache. The other half of :meth:`reset`."""
        ...


@dataclass(slots=True)
class VLAConfig:
    """Load-time + planning configuration for :class:`VLAEngine`.

    Defaults track OpenVLA-7B: 7-DoF end-effector actions, single-step decode,
    256-bin discretization. Chunked VLAs (π0-FAST, GR00T-class) raise
    ``action_chunk``; the seam is unchanged.
    """

    repo: str = DEFAULT_REPO
    device_index: int = 0
    dtype: str = "bfloat16"
    # Textual instruction / goal conditioning (the VLM's language prompt).
    prompt: str | None = None
    # Executed/decoded action-vector width (7 = 6-DoF pose delta + 1 gripper).
    action_dim: int = 7
    # Actions decoded per step. OpenVLA = 1; chunked VLAs decode K > 1.
    action_chunk: int = 1
    # Per-dim action-token discretization bins (sizes the Phase-1 detokenizer).
    action_bins: int = 256
    # CEM/candidate planning knobs, used by ``plan()``.
    plan_candidates: int = 8  # action-token sequences decoded per planning call
    # Decode all candidates in one batched forward (shared prefix KV) instead of
    # a per-candidate loop, when the pipeline advertises ``supports_batch``. This
    # is the token-VLA analogue of ``VJepa2ACConfig.plan_batched``.
    plan_batched: bool = True
    # Wire-facing control-space tag on emitted actions. ``None`` derives
    # ``vla_{action_dim}d`` (e.g. ``vla_7d``).
    space: str | None = None


@dataclass(slots=True)
class _Session:
    """Engine-held per-session state the wire-safe ``WorldState`` cannot carry."""

    pending_actions: torch.Tensor | None = field(default=None)


class VLAEngine:
    """A token-decoding VLA served on a Repercep backend (the interactive seam).

    The action loop (advance → decode) and session bookkeeping are implemented
    model-agnostically against an injected :class:`_VLAPipeline`; in production
    the pipeline is built from the HF bundle on the first :meth:`load` (the
    Phase-1 port), in tests it is a fake. ``WorldState.context`` carries the
    running decode context ``(T, D)``; the KV cache lives in the pipeline keyed
    by ``session_id``.
    """

    model_name = "vla"

    def __init__(
        self,
        backend: Backend,
        config: VLAConfig | None = None,
        *,
        pipeline: _VLAPipeline | None = None,
    ) -> None:
        self._backend = backend
        self._config = config if config is not None else VLAConfig()
        # Injected for testing / advanced use; otherwise loaded by ``load()``.
        self._pipeline: _VLAPipeline | None = pipeline
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

    # --- loading (the model-specific port, Phase 1) ---

    def load(self) -> None:
        """Build the real pipeline (GPU: needs the bundle + transformers).

        The model-specific half lives in :mod:`repercep.models.vla_pipeline`
        (Phase 1): the processor, the action detokenizer, and the KV-cached
        autoregressive decode over the VLM backbone. It raises with the setup
        recipe when the dependency or checkpoint is missing. Idempotent.
        """
        if self._pipeline is not None:
            return
        from repercep.models.vla_pipeline import build_pipeline

        self._pipeline = build_pipeline(self._backend, self._config)

    # --- the interactive seam ---

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Open a session: build the KV cache, encode the seed observation."""
        pipeline = self._require_pipeline()
        session_id = uuid.uuid4().hex
        pipeline.reset(session_id, self._config.prompt)
        context = pipeline.encode_observation(session_id, conditioning)
        self._sessions[session_id] = _Session()
        return WorldState(context=context, step_index=0, session_id=session_id)

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one step under the *executed* action chunk.

        Pushes the executed action into the KV context, then greedily decodes
        the next proposed chunk (parked for a following :meth:`plan`).
        ``action.values`` is a flat executed chunk — ``k x action_dim`` floats.
        """
        pipeline = self._require_pipeline()
        session = self._session_for(state)
        executed = self._parse_action_chunk(action, like=state.context)
        new_context = pipeline.append_executed(state.session_id, executed, state.context)
        proposed = pipeline.decode_action_chunk(state.session_id, new_context, 1)
        session.pending_actions = proposed[0]
        new_state = WorldState(
            context=new_context, step_index=state.step_index + 1, session_id=state.session_id
        )
        return new_state, LatentStep(step_index=new_state.step_index)

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Candidate-batched planning: decode N action-token sequences, score
        them against ``goal``, return the first action of the lowest-cost one.

        ``horizon`` is accepted for seam compatibility; the decoded chunk length
        is ``action_chunk`` (v0 scores single chunks — multi-chunk rollout is
        Phase-1, recorded in the port plan). The candidate set shares the prompt
        +observation prefix KV, so batching them through one decode (the
        ``plan_batched`` lever) is the measured serving win, exactly as for
        V-JEPA 2-AC's CEM candidates.
        """
        import torch

        pipeline = self._require_pipeline()
        self._session_for(state)  # validate the state belongs to this engine
        n = self._config.plan_candidates
        batched = self._config.plan_batched and getattr(pipeline, "supports_batch", False)
        if batched:
            candidates = pipeline.decode_action_chunk(state.session_id, state.context, n)
        else:
            rows = [
                pipeline.decode_action_chunk(state.session_id, state.context, 1)[0]
                for _ in range(n)
            ]
            candidates = torch.stack(rows)
        costs = pipeline.score_candidates(state.session_id, candidates, goal)
        best = int(torch.argmin(costs).item())
        chosen = candidates[best]
        self._session_for(state).pending_actions = chosen
        return Action(values=chosen[0].tolist(), space=self._wire_space())

    def release(self, state: WorldState) -> None:
        """Drop this session's server-side state (KV cache + bookkeeping).

        Not part of :class:`InteractiveWorldModel` — the serving layer calls it
        duck-typed when a client session ends. Both layers hold per-session
        state (this engine's ``_sessions`` and the pipeline's KV cache), so both
        need dropping or the cache leaks one entry per session under churn.
        """
        self._sessions.pop(state.session_id, None)
        if self._pipeline is not None:
            self._pipeline.close(state.session_id)

    # --- internals ---

    def _require_pipeline(self) -> _VLAPipeline:
        if self._pipeline is None:
            self.load()
        assert self._pipeline is not None
        return self._pipeline

    def _session_for(self, state: WorldState) -> _Session:
        session = self._sessions.get(state.session_id)
        if session is None:
            raise KeyError(
                f"unknown session {state.session_id!r} — WorldState must come "
                "from this engine's reset()"
            )
        return session

    def _wire_space(self) -> str:
        return self._config.space or f"vla_{self._config.action_dim}d"

    def _parse_action_chunk(self, action: Action, *, like: torch.Tensor) -> torch.Tensor:
        """Validate + shape a flat executed chunk to ``(k, action_dim)``."""
        import torch

        a_dim = self._config.action_dim
        if len(action.values) % a_dim != 0:
            raise ValueError(
                f"action chunk length {len(action.values)} is not a multiple "
                f"of action_dim={a_dim} (space={action.space!r})"
            )
        vec = torch.tensor(action.values, dtype=torch.float32, device=like.device)
        return vec.reshape(-1, a_dim)
