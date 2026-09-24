"""V-JEPA 2-AC engine — the lead interactive, energy-based world model.

V-JEPA 2 (Meta, 2025) is an encoder-predictor video world model from Yann
LeCun's group: it predicts in *representation space*, not pixels, and is the
flagship of the energy-based / JEPA family. The action-conditioned variant
(V-JEPA 2-AC, ~300M params, block-causal) autoregressively predicts the next
state embedding conditioned on an action, and *plans* by minimizing a
latent-space energy (embedding distance to a goal) over candidate action
sequences — model-predictive control as energy minimization.

This module wraps it as an
:class:`~repercep.runtime.interactive.InteractiveWorldModel` on a Repercep
``Backend``. What is **implemented and tested** here is the model-agnostic
algorithmic layer: the latent rollout (:meth:`VJepa2ACEngine.step`), the
energy function (:meth:`VJepa2ACEngine._rollout_energy`), and the CEM/MPC
planner (:meth:`VJepa2ACEngine._plan_sequence` / :meth:`VJepa2ACEngine.plan`).
They run against an injected ``encoder`` + ``predictor`` (see ``__init__``), so
the loop and the planner are exercised on CPU without any model weights.

The model-specific weight loading is now wired (ADR-0008 Phase 1): the encoder is
the same HuggingFace checkpoint ``scripts/run_vjepa2.py`` runs
(``facebook/vjepa2-vitg-fpc64-256``, loaded in :meth:`VJepa2ACEngine._ensure_encoder`);
the AC predictor head — which is *not* an HF ``AutoModel`` — is pulled from
``facebookresearch/vjepa2`` via its Torch Hub entrypoint and wrapped by
:class:`_AcPredictorAdapter` (:func:`_load_ac_predictor`). Because the real
predictor works over **patch tokens** and predicts a whole next frame (not the
single per-frame embedding the stub uses), the engine is patch-aware via
``_tokens_per_frame`` (1 on the stub path, ``(crop/patch)**2`` on the real one).
These real paths need a GPU + the checkpoints to run; the spots that have no
offline ground truth are marked ``VERIFY ON GPU`` in :class:`_AcPredictorAdapter`.
Real image/video URI decode is :func:`_decode_observation` (serving IO, Phase 2).
See ``docs/adr/0008-interactive-world-model-seam.md``.

This is NOT a diffusion model: there is no denoise loop, so the adaptive cache
does not apply. What carries over from the Cosmos/Wan path is the ``Backend``
seam (ADR-0003), the attention abstraction, the config + ``REPERCEP_*``
conventions, the ``EngineInfo`` contract, and ``torch.inference_mode()``
discipline (the F18 lesson).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from repercep.runtime.engine import EngineInfo
from repercep.runtime.types import Action, ConditioningKind, LatentStep, WorldState

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend
    from repercep.runtime.types import ConditioningInput, RolloutParams

#: The V-JEPA 2 encoder checkpoint (the one ``scripts/run_vjepa2.py`` benchmarks).
DEFAULT_ENCODER_REPO = "facebook/vjepa2-vitg-fpc64-256"

#: The action-conditioned predictor head. NOTE: this lives in the
#: ``facebookresearch/vjepa2`` research repo today, not as an HF ``AutoModel`` —
#: the exact checkpoint + loader is a Phase-1 port item (see ADR-0008).
DEFAULT_PREDICTOR_REPO = "facebookresearch/vjepa2"


class _Predictor(Protocol):
    """The action-conditioned next-state predictor, as a callable.

    Maps the current latent context window and an action to the next state
    embedding. Keeping this an injectable callable is what isolates the
    model-specific port from the model-agnostic rollout/planning logic.
    """

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...


class _CachedPredictor(Protocol):
    """Optional incremental (KV-cache) mode for :class:`_Predictor`.

    ``cache`` is opaque to the engine (predictor-owned per-layer K/V, however
    it wants to represent them); the engine only bootstraps it, advances it,
    appends to it, and (policy-gated) evicts from it. See
    ``docs/adr/0009-kv-latent-reuse.md`` for the two GPU-verified facts that
    shape this contract, neither obvious from the model's architecture alone:

    1. This model's RoPE is *not* a composable rotation (a maintainer-
       acknowledged bug in the upstream frequency tiling) — a sliding-window
       evict is NOT a cheap shift. Positions must be re-derived fresh at the
       correct window offset on every use; caching K **pre-rotation** is what
       makes that possible without a recompute.
    2. Eviction is structurally unrecoverable from a K/V cache alone for a
       full-depth causal transformer (layer 1+ hidden states are already
       contaminated by attending the evicted frame at layer 0). ``evict`` is
       therefore an explicit, disclosed approximation — only called when the
       engine's ``kv_evict_policy`` is ``"slide"``; the default ``"reinit"``
       policy never calls it (see :class:`VJepa2ACConfig`).

    ``step_cached`` predicts only — it does not durably append its own
    output. The engine layer-norms the prediction (:meth:`VJepa2ACEngine._maybe_norm`)
    before it re-enters the context, so only the engine has the block that's
    actually valid to cache; it hands that back via :meth:`append_frame`.
    """

    supports_kv_cache: bool

    def init_cache(self, context: torch.Tensor) -> Any:
        """Bootstrap a cache from a full context window (pays the full-window
        forward cost once; every subsequent :meth:`step_cached` call doesn't)."""
        ...

    def step_cached(self, cache: Any, action: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Predict the next frame from ``cache`` + ``action``. Returns the
        predicted next frame (same contract as ``_Predictor.__call__``'s
        return) and a (possibly unchanged) ``cache`` — the engine only ever
        feeds this returned ``cache`` on to :meth:`append_frame`, never
        directly to another :meth:`step_cached` call."""
        ...

    def append_frame(self, cache: Any, normed_block: torch.Tensor) -> Any:
        """Durably append the engine-normed newest frame to ``cache``.

        ``normed_block`` is exactly what :meth:`VJepa2ACEngine.step` is about
        to append to ``WorldState.context`` — the source of truth for what
        "the newest cached frame" means."""
        ...

    def evict(self, cache: Any, sink_frames: int = 0) -> Any:
        """Drop the oldest cached frame, keeping the first ``sink_frames``
        frames exempt (StreamingLLM-style attention sinks). NOT a free/exact
        operation for a full-depth causal transformer (see the class
        docstring, fact 2) — only called under ``kv_evict_policy="slide"``,
        an explicit accepted approximation, never under the default
        ``"reinit"`` policy."""
        ...


@dataclass(slots=True)
class _CacheEntry:
    """A session's KV cache plus the context it is synchronized to.

    ``last_context`` is what makes cache reuse safe: :meth:`VJepa2ACEngine.step`
    only trusts ``cache`` when the incoming ``WorldState.context`` matches it
    exactly. A CEM rollout (:meth:`VJepa2ACEngine._rollout_energy`) restarts
    every candidate from the *same* starting ``state`` under the *same*
    ``session_id`` — without this check, candidate 2+ would silently resume
    from candidate 1's cache (wrong context, corrupted energies). A mismatch
    means "this call didn't continue the last cached step" and falls back to
    a fresh :meth:`_CachedPredictor.init_cache` — the same cost as no cache,
    never a wrong answer, and (since a rollout's steps *after* the first one
    within one candidate DO match) still a real speedup for that path.
    """

    cache: Any
    last_context: torch.Tensor


@dataclass(slots=True)
class _AcKVCache:
    """Real AC-predictor cache: zero-action prefix K/V plus one pending frame.

    The public adapter's full forward supplies zeros for every historical
    action and the requested action only at the trailing frame.  Persisting
    the action-conditioned trailing K/V would therefore change the model on
    the next step.  ``layer_kv`` contains only frames processed with zero
    actions; ``pending`` is the newest visual block, processed with the
    caller's action by :meth:`_AcPredictorAdapter.step_cached` but not made
    durable until a separate zero-action pass has produced parity-safe K/V.
    """

    layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    processed_frames: int
    pending: torch.Tensor | None


@dataclass(slots=True)
class VJepa2ACConfig:
    """Load-time + planning configuration for :class:`VJepa2ACEngine`."""

    encoder_repo: str = DEFAULT_ENCODER_REPO
    predictor_repo: str = DEFAULT_PREDICTOR_REPO
    device_index: int = 0
    dtype: str = "bfloat16"
    # Block-causal window the predictor attends over (state embeddings retained).
    context_frames: int = 8
    # Synthetic-seed clip shape for an unconditioned (NONE) reset. Real
    # image/video URI decode is Phase 2 (serving IO).
    seed_frames: int = 64
    seed_resolution: int = 256
    # Energy-based planning (CEM / MPC) knobs, used by ``plan()``.
    plan_samples: int = 64  # candidate action sequences sampled per planning call
    plan_elites: int = 8  # top-k by lowest energy, refit each iteration
    plan_iters: int = 3  # CEM refit iterations
    action_dim: int = 7  # control dimensionality (e.g. 7-DoF end-effector delta)
    # Receding-horizon warm start: seed each plan() from the previous solution
    # for the session, shifted one step (standard MPC shift-reuse). First call
    # and horizon changes fall back to the zero mean, so cold behavior is
    # unchanged. Latency lever (c) of the 2026-07 plan.
    plan_warm_start: bool = True
    # Batch all CEM candidates through the predictor as one forward per rollout
    # timestep instead of the per-candidate loop (measured 1.6x H100 / 2.1x
    # MI300X in scripts/bench_cem_batched.py). Used only when the predictor
    # advertises ``supports_batch``; injected single-sample stubs keep the loop.
    plan_batched: bool = True
    # Dtype the AC predictor computes in. ``float32`` is the conservative
    # default (the upstream RoPE attention upcasts q/k to fp32, so bf16 weights
    # hit an SDPA dtype mismatch). ``bfloat16`` is the fast path: the adapter
    # harmonizes q/k back to v.dtype at the SDPA boundary
    # (:func:`_sdpa_dtype_harmonizer`). GPU-verified 2026-07-11 (H100): energy
    # parity with fp32 to bf16 resolution, 4.3x on the batched plan
    # (``docs/LEVERS_2026_07_H100.md``).
    predictor_compute_dtype: str = "float32"
    # Incremental KV-cache path for step() (docs/adr/0009-kv-latent-reuse.md):
    # only new-frame tokens attend against a persisted cache instead of
    # re-encoding the whole context window every step. Used only when the
    # predictor advertises ``supports_kv_cache``.
    use_kv_cache: bool = True
    # How many frames `reset()` seeds the window with, decoupled from the max
    # attention window (`context_frames`). None (default) reproduces today's
    # behavior: the seed clip is sliced straight to `context_frames`, so the
    # window starts already at cap. Setting this below `context_frames` (e.g.
    # 8) is what makes the *growing*-window regime (exact, GPU-verified)
    # actually occur: steps 1..(context_frames - reset_context_frames) are
    # pure cache growth, no eviction, bit-exact vs a full recompute.
    reset_context_frames: int | None = None
    # What happens once the window is full and the next step would evict the
    # oldest cached frame. "reinit" (default, exact): drop the session's
    # cache; subsequent saturated steps use the ordinary full predictor — the
    # same answer and cost as use_kv_cache=False, never wrong. Because a
    # strictly-capped window evicts on *every* step once saturated, "reinit"
    # only wins for sessions/episodes bounded by context_frames -
    # reset_context_frames steps of pure growth; steps past saturation cost
    # the same as no cache at all (graceful degradation, not a silent
    # approximation). "slide": call the predictor's `evict` —
    # an explicit, disclosed approximation (ADR-0009 Finding 2: not exact for
    # a full-depth causal transformer) — only enable after running
    # scripts/verify_kv_regimes.py and accepting its measured drift.
    kv_evict_policy: str = "reinit"  # "reinit" | "slide"
    # Attention-sink prefix length for the "slide" policy (StreamingLLM-style
    # kept-forever frames), ignored under "reinit". 0 = no sink.
    kv_sink_frames: int = 0
    # KV-cache the CEM-batched plan() path itself (ADR-0009's deferred second
    # reuse dimension: the S candidates in one _rollout_energy_batched call
    # share the pre-rollout context prefix, currently recomputed S times).
    # Requires use_kv_cache=True and a predictor with supports_kv_cache; the
    # growing-window precondition applies to the WHOLE horizon (see
    # _rollout_energy_batched_cached) — falls back to the plain batched path
    # whenever it doesn't hold, so this is never wrong, only sometimes not
    # faster. Default False: GPU-verify is pending (CPU-parity-tested only so
    # far), same discipline as the persistent-session path before its own
    # verify landed.
    plan_batched_kv: bool = False


class VJepa2ACEngine:
    """V-JEPA 2-AC served on a Repercep backend (the interactive seam).

    The rollout (:meth:`step`), the energy (:meth:`_rollout_energy`), and the
    CEM planner (:meth:`plan`) are implemented model-agnostically against an
    ``encoder`` + ``predictor``. In production those are loaded lazily on the
    first :meth:`load`; in tests they are injected so the loop and planner run
    on CPU without weights. The weight loaders are the remaining port (see the
    module docstring and ADR-0008).
    """

    model_name = "vjepa2-ac-300m"

    def __init__(
        self,
        backend: Backend,
        config: VJepa2ACConfig | None = None,
        *,
        encoder: Any | None = None,
        predictor: _Predictor | None = None,
    ) -> None:
        self._backend = backend
        self._config = config if config is not None else VJepa2ACConfig()
        # Injected for testing / advanced use; otherwise loaded lazily.
        self._encoder: Any | None = encoder
        self._predictor: _Predictor | None = predictor
        # Spatial patch tokens per temporal frame in the encoder's output. The
        # real V-JEPA 2 encoder emits ``(crop/patch)**2`` patch tokens per frame;
        # the model-agnostic stub path (injected encoder, no ``config``) keeps the
        # default of 1, so ``WorldState.context`` rows == frames for the tests.
        # Set from the encoder config in :meth:`_ensure_encoder` on the real path.
        self._tokens_per_frame = 1
        # The real V-JEPA 2-AC predictor is trained on layer-normed reps
        # (``normalize_reps=True`` in the reference wrapper). Enabled on the real
        # path in :meth:`_ensure_encoder`; left off for the model-agnostic stub.
        self._normalize_reps = False
        # Per-session previous plan solution, for receding-horizon warm start.
        self._plan_mean: dict[str, torch.Tensor] = {}
        # Per-session incremental predictor cache (see _CacheEntry / step()).
        self._kv_cache: dict[str, _CacheEntry] = {}

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None and self._predictor is not None

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name=self.model_name,
            backend=self._backend.name,
            device=f"{self._backend.name}:{self._config.device_index}",
            dtype=self._config.dtype,
            ready=self.is_loaded,
        )

    # --- loading (the model-specific port) ---

    def _ensure_encoder(self) -> None:
        """Load the V-JEPA 2 encoder if not already present/injected.

        The proven path from ``scripts/run_vjepa2.py``: the HF ``AutoModel`` whose
        ``get_vision_features`` :meth:`reset` calls. The AC predictor's encoder is
        frozen during action-conditioned post-training, so this same checkpoint is
        the one the predictor expects embeddings from.
        """
        if self._encoder is not None:
            return
        import torch
        from transformers import AutoModel

        device = self._backend.torch_device(self._config.device_index)
        dtype = getattr(torch, self._config.dtype)
        self._encoder = (
            AutoModel.from_pretrained(self._config.encoder_repo, dtype=dtype).to(device).eval()
        )
        self._tokens_per_frame = _infer_tokens_per_frame(self._encoder)
        self._normalize_reps = True

    def _ensure_predictor(self) -> None:
        """Load the action-conditioned predictor head if not already present."""
        if self._predictor is not None:
            return
        # The encoder sets ``_tokens_per_frame`` (the spatial grid the predictor's
        # next-frame block spans), so load it first.
        self._ensure_encoder()
        self._predictor = _load_ac_predictor(self._config, self._tokens_per_frame)

    def load(self) -> None:
        """Load the encoder and the AC predictor. Idempotent."""
        self._ensure_encoder()
        self._ensure_predictor()

    # --- the interactive seam (implemented) ---

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Encode the conditioning observation into the initial world state."""
        import torch

        self._ensure_encoder()
        assert self._encoder is not None
        frames = self._resolve_frames(conditioning)
        with torch.inference_mode():
            features = self._encoder.get_vision_features(pixel_values_videos=frames)
        # Keep the last N *frames*; each frame is ``_tokens_per_frame``
        # patch-token rows (1 on the stub path). N is ``reset_context_frames``
        # when set (below the ``context_frames`` cap, so the window *grows*
        # into the cap over the next several steps — the exact KV-cache
        # regime, ADR-0009); otherwise ``context_frames`` (today's default:
        # the window starts already at cap, so a wired cache would evict
        # from step 1).
        seed_frames = self._config.reset_context_frames or self._config.context_frames
        keep = min(seed_frames, self._config.context_frames) * self._tokens_per_frame
        context = self._maybe_norm(_as_context(features)[-keep:])
        return WorldState(context=context, step_index=0, session_id=_new_session_id())

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one latent step under ``action`` (no pixel decode)."""
        import torch

        self._ensure_predictor()
        assert self._predictor is not None
        cache_capable = self._config.use_kv_cache and getattr(
            self._predictor, "supports_kv_cache", False
        )
        frames_now = int(state.context.shape[0]) // self._tokens_per_frame
        entry = self._kv_cache.get(state.session_id)
        cache_reusable = entry is not None and _same_tensor(entry.last_context, state.context)
        # In exact "reinit" mode a saturated window cannot produce a reusable
        # next cache.  If no matching growth-phase cache remains, call the
        # ordinary predictor directly instead of paying init_cache() overhead
        # only to discard the result after this step.
        cached_path = cache_capable and (
            self._config.kv_evict_policy == "slide"
            or frames_now < self._config.context_frames
            or cache_reusable
        )
        if cache_capable and not cached_path:
            self._kv_cache.pop(state.session_id, None)
        with torch.inference_mode():
            vec = torch.tensor(
                action.values, dtype=state.context.dtype, device=state.context.device
            )
            cache: Any = None
            if cached_path:
                nxt, cache = self._step_cached(state, vec)
            else:
                nxt = self._predictor(state.context, vec)
            # The predictor returns the next frame: a single embedding ``(D,)`` on
            # the stub path, or a ``(P, D)`` block of patch tokens on the real
            # path. Shape to a 2-D block, (real-path) layer-norm it like the
            # reference wrapper, append, then cap to the window.
            block = self._maybe_norm(nxt if nxt.ndim == 2 else nxt.unsqueeze(0))
            context = torch.cat([state.context, block], dim=0)
            keep = self._config.context_frames * self._tokens_per_frame
            context = context[-keep:]
        new_state = WorldState(
            context=context, step_index=state.step_index + 1, session_id=state.session_id
        )
        if cached_path:
            self._sync_kv_cache(state, new_state.session_id, context, cache, block)
        return new_state, LatentStep(step_index=new_state.step_index)

    def _step_cached(self, state: WorldState, vec: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """The incremental path: reuse the session's cache when it's in sync.

        "In sync" means the incoming ``state.context`` is exactly what the
        cache last produced (see :class:`_CacheEntry`) — true for real
        sequential advancement (the common case) and for every step after the
        first within one CEM candidate's rollout; false (falls back to a full
        :meth:`_CachedPredictor.init_cache`, same cost as no cache) whenever a
        rollout branches to a different starting state under the same session.

        Only *predicts* — the returned ``cache`` is not yet durably advanced;
        :meth:`_sync_kv_cache` does that once ``step()`` has the engine-normed
        block, per the class docstring on :class:`_CachedPredictor`.
        """

        # Only reached when getattr(..., "supports_kv_cache", False) is true
        # (checked by both call sites), so the real object satisfies
        # _CachedPredictor even though the injected-callable _Predictor type
        # doesn't declare it.
        predictor = cast("_CachedPredictor", self._predictor)
        entry = self._kv_cache.get(state.session_id)
        if entry is not None and _same_tensor(entry.last_context, state.context):
            cache = entry.cache
        else:
            cache = predictor.init_cache(state.context)
        next_frame, cache = predictor.step_cached(cache, vec)
        return next_frame, cache

    def _sync_kv_cache(
        self,
        state: WorldState,
        session_id: str,
        new_context: torch.Tensor,
        cache: Any,
        normed_block: torch.Tensor,
    ) -> None:
        """Durably append the predicted frame and apply the eviction policy.

        ``state`` is the *incoming* (pre-append) state — its frame count is
        what decides whether this step is sliding the window (see
        :class:`VJepa2ACConfig`'s ``kv_evict_policy`` docstring for why the
        pre-append count, not ``new_context``'s already-capped one, is the
        right signal).
        """
        predictor = cast("_CachedPredictor", self._predictor)
        frames_now = int(state.context.shape[0]) // self._tokens_per_frame
        window_full = frames_now >= self._config.context_frames
        if window_full and self._config.kv_evict_policy == "reinit":
            # Exact: don't touch a cache we're not confident is sound to
            # evict from. Drop it; the next call uses the full predictor.
            self._kv_cache.pop(session_id, None)
            return
        cache = predictor.append_frame(cache, normed_block)
        if window_full and self._config.kv_evict_policy == "slide":
            cache = predictor.evict(cache, self._config.kv_sink_frames)
        self._kv_cache[session_id] = _CacheEntry(cache=cache, last_context=new_context)

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Energy-minimizing MPC (CEM): the next action toward ``goal``.

        Samples action sequences, rolls each out via :meth:`step`, scores them
        by the terminal latent energy, refits to the elite set, and returns the
        first action of the energy-minimizing sequence.
        """
        sequence = self._plan_sequence(state, goal, horizon)
        return Action(values=sequence[0].tolist(), space="ee_delta")

    def release(self, state: WorldState) -> None:
        """Drop this session's server-side state (KV cache, warm-start mean).

        Not part of :class:`InteractiveWorldModel` — the serving layer calls
        it duck-typed (``getattr(engine, "release", None)``) when a client
        session ends, so engines with no per-session state to free (or that
        predate this method) don't need a no-op override. Without this, a
        churn of short-lived sessions leaks one GPU KV-cache entry per
        session forever, since ``step()`` only ever *adds* to ``_kv_cache``/
        ``_plan_mean`` and nothing previously removed an entry on session
        end (only mid-session, when a saturated "reinit" cache is dropped —
        see ``step()``).
        """
        self._kv_cache.pop(state.session_id, None)
        self._plan_mean.pop(state.session_id, None)

    # --- planning internals (energy-based) ---

    def _rollout_energy(
        self, state: WorldState, action_seq: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """Energy of a rollout: distance of its terminal state to ``goal``.

        This is the scalar the planner minimizes — low energy means the action
        sequence drives the world model toward the goal embedding.
        """
        import torch

        rolled = state
        for t in range(int(action_seq.shape[0])):
            rolled, _ = self.step(rolled, Action(values=action_seq[t].tolist()))
        # Terminal state = the last frame (its ``_tokens_per_frame`` rows; the last
        # single row on the stub path). ``vector_norm`` flattens, so the stub case
        # ``[-1:]`` is numerically identical to the previous ``[-1]``.
        terminal = rolled.context[-self._tokens_per_frame :]
        energy: torch.Tensor = torch.linalg.vector_norm(terminal - goal)
        return energy

    def _plan_sequence(self, state: WorldState, goal: torch.Tensor, horizon: int) -> torch.Tensor:
        """Cross-entropy-method search for the energy-minimizing action sequence."""
        import torch

        mean = self._warm_start_mean(state, goal, horizon)
        std = torch.ones_like(mean)
        for _ in range(self._config.plan_iters):
            noise = torch.randn(
                self._config.plan_samples,
                horizon,
                self._config.action_dim,
                dtype=goal.dtype,
                device=goal.device,
            )
            seqs = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            energies = self._candidate_energies(state, seqs, goal)
            elite_idx = torch.topk(energies, self._config.plan_elites, largest=False).indices
            elite = seqs[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp_min(1e-6)
        if self._config.plan_warm_start:
            self._plan_mean[state.session_id] = mean.detach()
        return mean

    def _warm_start_mean(self, state: WorldState, goal: torch.Tensor, horizon: int) -> torch.Tensor:
        """Initial CEM mean: the previous solution shifted one step, else zeros.

        Receding-horizon shift-reuse: after executing the first action of the
        last plan, its remaining tail is the best-known guess for this step's
        prefix (the final row is repeated to fill the horizon). Falls back to
        the zero mean on the first call for a session or a horizon change.
        """
        import torch

        prev = self._plan_mean.get(state.session_id) if self._config.plan_warm_start else None
        if prev is not None and tuple(prev.shape) == (horizon, self._config.action_dim):
            shifted = torch.cat([prev[1:], prev[-1:]], dim=0)
            return shifted.to(dtype=goal.dtype, device=goal.device)
        return torch.zeros(horizon, self._config.action_dim, dtype=goal.dtype, device=goal.device)

    def _candidate_energies(
        self, state: WorldState, seqs: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """Energies of ``(S, H, A)`` candidate sequences, batched when possible.

        The batched path rolls all candidates through the predictor as one
        forward per timestep (latency lever (b), measured 1.6-2.1x); it needs a
        predictor that accepts a batch, which the real
        :class:`_AcPredictorAdapter` advertises via ``supports_batch``.
        Injected single-sample stubs (and ``plan_batched=False``) keep the
        per-candidate loop, so the model-agnostic tests are unaffected.
        """
        import torch

        if self._config.plan_batched and getattr(self._predictor, "supports_batch", False):
            if self._config.plan_batched_kv and self._config.use_kv_cache:
                cached = self._rollout_energy_batched_cached(state, seqs, goal)
                if cached is not None:
                    return cached
            return self._rollout_energy_batched(state, seqs, goal)
        return torch.stack(
            [self._rollout_energy(state, seqs[i], goal) for i in range(int(seqs.shape[0]))]
        )

    def _rollout_energy_batched(
        self, state: WorldState, seqs: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """Terminal energies of all ``(S, H, A)`` candidates in one rollout.

        Shares the context prefix across candidates: the window is expanded to
        the batch and each timestep is a single batched predictor forward,
        numerically identical to looping :meth:`_rollout_energy` per candidate
        (same append-and-cap window, same norm — asserted by the parity test).
        """
        import torch

        self._ensure_predictor()
        assert self._predictor is not None
        n_candidates = int(seqs.shape[0])
        keep = self._config.context_frames * self._tokens_per_frame
        with torch.inference_mode():
            ctx = state.context.unsqueeze(0).expand(n_candidates, -1, -1).contiguous()
            for t in range(int(seqs.shape[1])):
                blocks = self._predictor(ctx, seqs[:, t])  # (S, P, D)
                ctx = torch.cat([ctx, self._maybe_norm(blocks)], dim=1)[:, -keep:]
            terminal = ctx[:, -self._tokens_per_frame :]
            energies: torch.Tensor = torch.linalg.vector_norm(
                terminal - goal, dim=tuple(range(1, terminal.ndim))
            )
        return energies

    def _rollout_energy_batched_cached(
        self, state: WorldState, seqs: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor | None:
        """KV-cached terminal energies of all ``(S, H, A)`` candidates.

        ADR-0009's deferred CEM-batched reuse dimension: the pre-rollout
        context prefix is identical across all S candidates (block-causal —
        a frame's K/V never depends on a later frame's action token), so its
        K/V is computed **once** (:meth:`_CachedPredictor.init_cache`, batch
        1) instead of recomputed inside every batched forward. Candidates
        diverge from the first predicted frame on; from there each rollout
        step is one single-frame query per candidate against the shared
        cache (:meth:`_CachedPredictor.step_cached`, whose batch dimension
        the adapter expands from 1 up to S on first use — see
        ``_AcPredictorAdapter._advance_pending``), not a full O(window)
        re-encode of the growing context S times.

        Growing-window only, same soundness bound as the persistent-session
        path (ADR-0009 Finding 2: eviction is not exact) — but checked over
        the WHOLE horizon here, since all S candidates advance H steps
        together and every one of those steps must stay eviction-free.
        Returns ``None`` (never a wrong answer) when that doesn't hold, so
        the caller falls back to :meth:`_rollout_energy_batched` — the exact
        same fallback discipline ``step()`` uses for the single-candidate
        cache.
        """
        import torch

        self._ensure_predictor()
        assert self._predictor is not None
        if not getattr(self._predictor, "supports_kv_cache", False):
            return None
        predictor = cast("_CachedPredictor", self._predictor)
        horizon = int(seqs.shape[1])
        frames_now = int(state.context.shape[0]) // self._tokens_per_frame
        if frames_now + horizon > self._config.context_frames:
            return None

        with torch.inference_mode():
            cache = predictor.init_cache(state.context)
            terminal = None
            for t in range(horizon):
                predicted, cache = predictor.step_cached(cache, seqs[:, t])
                terminal = self._maybe_norm(predicted)
                cache = predictor.append_frame(cache, terminal)
        assert terminal is not None  # horizon >= 1 is guaranteed by plan()'s caller
        energies: torch.Tensor = torch.linalg.vector_norm(
            terminal - goal, dim=tuple(range(1, terminal.ndim))
        )
        return energies

    def _resolve_frames(self, conditioning: ConditioningInput) -> torch.Tensor:
        """Resolve a conditioning observation to a ``(1, T, C, H, W)`` pixel clip.

        ``NONE`` yields a synthetic seed clip (the encoder maps it to the prior);
        ``IMAGE``/``VIDEO`` decode the ``uri`` into a clip. The result is placed on
        the backend device + dtype so it feeds the (real) encoder directly.

        NOTE: the exact pixel normalization the V-JEPA 2 encoder expects is the HF
        ``VJEPA2VideoProcessor`` recipe (resize/center-crop + ImageNet mean/std).
        ``_decode_observation`` applies that recipe; parity with the processor is a
        thing to confirm against real inputs on the GPU box (serving IO, Phase 2).
        """
        n, r = self._config.seed_frames, self._config.seed_resolution
        if conditioning.kind is ConditioningKind.NONE:
            import torch

            clip = torch.zeros(1, n, 3, r, r)
        else:
            clip = _decode_observation(conditioning, frames=n, resolution=r)
        return self._to_backend(clip)

    def _to_backend(self, clip: torch.Tensor) -> torch.Tensor:
        """Move a pixel clip onto the backend device + configured dtype.

        Guarded so the model-agnostic stub backend (no ``torch_device``) leaves the
        clip on CPU/float32 — the injected fake encoder ignores it anyway.
        """
        import torch

        to_device = getattr(self._backend, "torch_device", None)
        if to_device is None:
            return clip
        device = to_device(self._config.device_index)
        dtype = getattr(torch, self._config.dtype)
        return clip.to(device=device, dtype=dtype)

    def _maybe_norm(self, reps: torch.Tensor) -> torch.Tensor:
        """Layer-norm reps over the embedding dim on the real path (no-op on stub).

        Matches the reference ``WorldModel(normalize_reps=True)``: the AC predictor
        is trained on layer-normed reps, so both the seed context and each predicted
        frame are normalized.
        """
        if not self._normalize_reps:
            return reps
        import torch.nn.functional as F  # noqa: N812

        return F.layer_norm(reps, (reps.shape[-1],))


class _AcPredictorAdapter:
    """Bridge the real V-JEPA 2-AC predictor to the ``_Predictor`` contract.

    The research predictor's forward (verified against
    ``facebookresearch/vjepa2`` ``src/models/ac_predictor.py`` and the reference
    ``notebooks/utils/world_model_wrapper.py``) is
    ``forward(x, actions, states, extrinsics=None) -> tokens`` over **patch
    tokens** ``x: (B, N_ctxt, D)`` (``N_ctxt`` = ``T`` frames x ``P`` spatial
    patches). It interleaves one action + one state token *per frame*, so
    ``actions`` and ``states`` are ``(B, T, action_dim)``; it predicts
    frame-causally and the **next frame** is the trailing ``P`` rows of the output
    (``predictor(...)[:, -tokens_per_frame:]`` in the reference wrapper).

    This adapter exposes Repercep's ``(context, action) -> next_frame`` contract: it
    batches the context, drives the *next* frame with ``action`` at the last frame
    position (earlier positions are no-op zeros), supplies a zero proprioceptive
    state, calls the predictor, and returns the predicted next frame's ``P`` rows
    for :meth:`VJepa2ACEngine.step` to append.

    REFINE: the reference integrates a real 7-DoF robot pose across steps
    (``compute_new_pose``); zeros here exercise the dynamics as a function of the
    action, which is what the load/rollout/planning smoke test needs. A real
    closed-loop robot deployment would thread the pose through ``WorldState``.
    """

    def __init__(
        self,
        predictor: Any,
        tokens_per_frame: int,
        action_dim: int,
        compute_dtype: Any = None,
    ) -> None:
        self._predictor = predictor
        self._p = tokens_per_frame
        self._adim = action_dim
        # The predictor runs in this dtype; context is cast in and the result cast
        # back. The upstream RoPE attention upcasts q/k to float32, so running the
        # predictor in float32 keeps q/k/v dtypes consistent (bf16 hits a SDPA
        # dtype-mismatch). ``None`` (the stub/unit-test path) does no casting.
        self._cdtype = compute_dtype

    #: The engine's batched CEM path (``_rollout_energy_batched``) keys off this.
    supports_batch = True
    #: The persistent-session path can reuse a zero-action prefix exactly.
    supports_kv_cache = True

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        import torch

        in_dtype = context.dtype
        # Single sample ``(N, D) + (A,)`` or a candidate batch ``(S, N, D) + (S, A)``.
        batched = context.ndim == 3
        x = context if batched else context.unsqueeze(0)
        if self._cdtype is not None:
            x = x.to(self._cdtype)
        n_frames = int(x.shape[1]) // self._p
        # Per-frame action/state tokens (B, T, action_dim); drive the next frame
        # with `action` at the last position, no-op (zero) for past frames.
        actions = x.new_zeros(int(x.shape[0]), n_frames, self._adim)
        actions[:, -1] = action.to(actions.dtype)
        states = torch.zeros_like(actions)
        with self._sdpa_guard():
            out = self._predictor(x, actions, states)
        tokens = out if isinstance(out, torch.Tensor) else out.last_hidden_state
        next_frames: torch.Tensor = tokens[:, -self._p :]  # predicted next frame(s)
        result = next_frames if batched else next_frames[0]
        return result.to(in_dtype)

    def init_cache(self, context: torch.Tensor) -> _AcKVCache:
        """Cache every completed frame and leave the trailing frame pending.

        A call predicts *from* the trailing context frame, so that frame must
        still be processed with the new action.  All earlier frames use the
        same zero historical actions as :meth:`__call__` and can be cached.
        """
        if context.ndim != 2 or int(context.shape[0]) % self._p:
            raise ValueError("AC KV cache expects context shaped (frames * patches, dim)")
        in_dtype = context.dtype
        x = context.unsqueeze(0)
        if self._cdtype is not None:
            x = x.to(self._cdtype)
        frames = int(x.shape[1]) // self._p
        if frames < 1:
            raise ValueError("AC KV cache requires at least one context frame")

        pending = x[:, -self._p :]
        prefix = x[:, : -self._p]
        if frames == 1:
            empty = tuple(
                (x.new_empty(1, 0, 0, 0), x.new_empty(1, 0, 0, 0))
                for _ in self._predictor.predictor_blocks
            )
            return _AcKVCache(empty, processed_frames=0, pending=pending.to(in_dtype))

        layer_kv = self._encode_zero_action_prefix(prefix)
        return _AcKVCache(layer_kv, processed_frames=frames - 1, pending=pending.to(in_dtype))

    def step_cached(
        self, cache: _AcKVCache, action: torch.Tensor
    ) -> tuple[torch.Tensor, _AcKVCache]:
        """Predict from the pending frame, then commit its zero-action K/V.

        The second incremental pass is intentional.  The uncached adapter
        zeros historical actions, so retaining the action-conditioned K/V
        would silently make cached multi-step rollouts a different model.
        Both passes touch only one frame of queries against the shared prefix.

        ``action`` may be ``(action_dim,)`` (single session) or ``(S,
        action_dim)`` (S CEM candidates sharing this cache — see
        :meth:`_advance_pending`); the batch dimension threads through
        automatically and ``predicted`` comes back shaped to match.
        """
        import torch

        if cache.pending is None:
            raise ValueError("AC KV cache has no pending frame to predict from")
        in_dtype = cache.pending.dtype
        predicted, _ = self._advance_pending(cache, action)
        zeros = torch.zeros_like(action)
        _unused, committed = self._advance_pending(cache, zeros)
        return predicted.to(in_dtype), committed

    def append_frame(self, cache: _AcKVCache, normed_block: torch.Tensor) -> _AcKVCache:
        """Append a predicted, normed frame as the cache's new pending frame.

        ``normed_block`` is ``(P, D)`` (single session) or ``(S, P, D)`` (S
        CEM candidates, matching :meth:`step_cached`'s batched ``predicted``).
        """
        if cache.pending is not None:
            raise ValueError("AC KV cache append requires a completed prediction step")
        batched = normed_block.ndim == 3
        p_dim = int(normed_block.shape[1] if batched else normed_block.shape[0])
        if normed_block.ndim not in (2, 3) or p_dim != self._p:
            raise ValueError("AC KV cache frame has the wrong patch-token shape")
        pending = normed_block if batched else normed_block.unsqueeze(0)
        return _AcKVCache(cache.layer_kv, cache.processed_frames, pending)

    def evict(self, cache: _AcKVCache, sink_frames: int = 0) -> _AcKVCache:
        """Approximate FIFO eviction of one completed frame's raw K/V.

        Raw pre-RoPE keys are retained, so all surviving positions are rotated
        fresh on their next use.  Deeper-layer residual contamination still
        makes this approximate, as documented in ADR-0009; the engine only
        calls it under the explicit ``slide`` policy.
        """
        import torch

        if sink_frames < 0 or sink_frames >= cache.processed_frames:
            raise ValueError("kv_sink_frames must leave a non-sink frame available to evict")
        block = self._action_tokens + self._p
        start = sink_frames * block
        stop = start + block
        layer_kv = tuple(
            (
                torch.cat([k[:, :, :start], k[:, :, stop:]], dim=2),
                torch.cat([v[:, :, :start], v[:, :, stop:]], dim=2),
            )
            for k, v in cache.layer_kv
        )
        return _AcKVCache(layer_kv, cache.processed_frames - 1, cache.pending)

    @property
    def _action_tokens(self) -> int:
        return 3 if bool(getattr(self._predictor, "use_extrinsics", False)) else 2

    def _encode_zero_action_prefix(
        self, visual: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Run the ordinary full prefix once while capturing pre-RoPE K/V."""
        import torch

        predictor = self._predictor
        frames = int(visual.shape[1]) // self._p
        x = predictor.predictor_embed(visual)
        actions = x.new_zeros(int(x.shape[0]), frames, self._adim)
        states = torch.zeros_like(actions)
        a = predictor.action_encoder(actions).unsqueeze(2)
        s = predictor.state_encoder(states).unsqueeze(2)
        x = x.view(int(x.shape[0]), frames, self._p, int(x.shape[-1]))
        if self._action_tokens == 3:
            extrinsics = actions[..., :-1]
            e = predictor.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)

        mask = predictor.attn_mask[: x.shape[1], : x.shape[1]].to(x.device, non_blocking=True)
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        with self._sdpa_guard():
            for block in predictor.predictor_blocks:
                _q, k, v = _ac_raw_qkv(block.attn, block.norm1(x))
                captured.append((k, v))
                x = block(
                    x,
                    mask=None,
                    attn_mask=mask,
                    T=frames,
                    H=predictor.grid_height,
                    W=predictor.grid_width,
                    action_tokens=self._action_tokens,
                )
        return tuple(captured)

    def _advance_pending(
        self, cache: _AcKVCache, action: torch.Tensor
    ) -> tuple[torch.Tensor, _AcKVCache]:
        """Run one pending frame against the cached prefix at every layer.

        ``action`` is ``(action_dim,)`` for a single session (the
        persistent-session path, unchanged behavior) or ``(S, action_dim)``
        for S CEM candidates sharing this cache's prefix and diverging from
        here (:meth:`VJepa2ACEngine._rollout_energy_batched_cached`). The
        cache's own batch dim starts at 1 (one shared prefix — see
        :meth:`init_cache`) and is **expanded, not copied**, up to S on
        first use by a batched caller; once a step has run with S
        candidates, ``layer_kv``/``pending`` are genuinely batch-S (each
        candidate's own committed K/V), so later expand calls become no-ops.
        """
        import torch
        import torch.nn.functional as F  # noqa: N812

        assert cache.pending is not None
        predictor = self._predictor
        batched = action.ndim == 2
        actions_in = action if batched else action.unsqueeze(0)  # (S, adim)
        s = int(actions_in.shape[0])

        visual = cache.pending
        if int(visual.shape[0]) == 1 and s > 1:
            visual = visual.expand(s, -1, -1)
        if self._cdtype is not None:
            visual = visual.to(self._cdtype)
        x = predictor.predictor_embed(visual)
        action_row = actions_in.to(dtype=x.dtype, device=x.device).unsqueeze(1)  # (S, 1, adim)
        state_row = torch.zeros_like(action_row)
        pieces = [predictor.action_encoder(action_row), predictor.state_encoder(state_row)]
        if self._action_tokens == 3:
            pieces.append(predictor.extrinsics_encoder(state_row[..., :-1]))
        pieces.append(x)
        x = torch.cat(pieces, dim=1)

        extended: list[tuple[torch.Tensor, torch.Tensor]] = []
        with self._sdpa_guard():
            for layer, block in enumerate(predictor.predictor_blocks):
                q, k_new, v_new = _ac_raw_qkv(block.attn, block.norm1(x))
                k_old, v_old = cache.layer_kv[layer]
                if cache.processed_frames == 0:
                    k_old = k_new[:, :, :0]
                    v_old = v_new[:, :, :0]
                elif int(k_old.shape[0]) == 1 and s > 1:
                    k_old = k_old.expand(s, -1, -1, -1)
                    v_old = v_old.expand(s, -1, -1, -1)
                q = _ac_rotate_augmented(
                    block.attn, q, 1, predictor.grid_height, predictor.grid_width,
                    self._action_tokens, frame_offset=cache.processed_frames,
                )
                k_all_raw = torch.cat([k_old, k_new], dim=2)
                v_all = torch.cat([v_old, v_new], dim=2)
                k_all = _ac_rotate_augmented(
                    block.attn, k_all_raw, cache.processed_frames + 1,
                    predictor.grid_height, predictor.grid_width, self._action_tokens,
                )
                y = F.scaled_dot_product_attention(
                    q, k_all, v_all,
                    dropout_p=block.attn.proj_drop_prob,
                    is_causal=block.attn.is_causal,
                )
                y = y.transpose(1, 2).reshape_as(x)
                y = block.attn.proj_drop(block.attn.proj(y))
                x = x + block.drop_path(y)
                x = x + block.drop_path(block.mlp(block.norm2(x)))
                extended.append((k_all_raw, v_all))

        visual_out = x[:, self._action_tokens :]
        visual_out = predictor.predictor_proj(predictor.predictor_norm(visual_out))
        if not batched:
            visual_out = visual_out[0]
        committed = _AcKVCache(tuple(extended), cache.processed_frames + 1, pending=None)
        return visual_out, committed

    def _sdpa_guard(self) -> Any:
        """The bf16 SDPA harmonizer when computing in bf16; a no-op otherwise."""
        import contextlib

        import torch

        if self._cdtype is torch.bfloat16:
            return _sdpa_dtype_harmonizer()
        return contextlib.nullcontext()


#: The published V-JEPA 2-AC checkpoint (encoder + predictor weights). NOTE: the
#: repo's Torch Hub entrypoint hard-codes a placeholder ``localhost`` URL, so we
#: build the model with ``pretrained=False`` and load these weights ourselves.
_AC_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"


def _load_ac_predictor(config: VJepa2ACConfig, tokens_per_frame: int) -> _Predictor:
    """Load the V-JEPA 2-AC predictor head as a ``_Predictor`` callable.

    Wires the action-conditioned head from ``facebookresearch/vjepa2`` (it is not
    an HF ``AutoModel`` today) via its Torch Hub entrypoint. The entrypoint's
    ``pretrained=True`` path points at a placeholder ``localhost`` checkpoint URL,
    so we build the architecture with ``pretrained=False`` and load the published
    checkpoint's ``predictor`` sub-state-dict ourselves (the same key-cleaning the
    repo applies). The encoder is reused from the HF load (frozen during AC
    post-training, so they match). The predictor is wrapped in
    :class:`_AcPredictorAdapter` to satisfy the ``(context, action) -> next_frame``
    contract the rollout/planner expect.
    """
    import torch

    device = config.device_index
    _, predictor = torch.hub.load(  # type: ignore[no-untyped-call]
        config.predictor_repo, "vjepa2_ac_vit_giant", pretrained=False
    )
    ckpt = torch.hub.load_state_dict_from_url(_AC_CHECKPOINT_URL, map_location="cpu")
    state_dict = {
        k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt["predictor"].items()
    }
    predictor.load_state_dict(state_dict)
    # ``float32`` (default) is the GPU-verified path: the upstream RoPE attention
    # upcasts q/k to float32, so bf16 weights hit an SDPA dtype mismatch.
    # ``bfloat16`` (opt-in, latency lever) relies on the adapter's SDPA dtype
    # harmonizer to cast q/k back down at the boundary — VERIFY ON GPU.
    cdtype = getattr(torch, config.predictor_compute_dtype)
    predictor = predictor.to(f"cuda:{device}", dtype=cdtype).eval()
    return _AcPredictorAdapter(predictor, tokens_per_frame, config.action_dim, cdtype)


def _ac_raw_qkv(attn: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project augmented AC tokens without applying the model's RoPE."""
    qkv = (
        attn.qkv(x)
        .unflatten(-1, (3, attn.num_heads, -1))
        .permute(2, 0, 3, 1, 4)
    )
    return qkv[0], qkv[1], qkv[2]


def _ac_bug_compatible_rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """The checkpoint-compatible upstream rotation, including its tiling bug."""
    import torch

    dim = int(x.shape[-1])
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos.to(dtype=x.dtype), omega)
    # Deliberately tile rather than repeat_interleave: pretrained V-JEPA 2-AC
    # was trained with this upstream behavior (facebookresearch/vjepa2#15).
    sin = freq.sin().squeeze(-1).repeat(1, 1, 1, 2)
    cos = freq.cos().squeeze(-1).repeat(1, 1, 1, 2)
    y1, y2 = x.reshape(*x.shape[:-1], dim // 2, 2).unbind(-1)
    rotated = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * cos) + (rotated * sin)


def _ac_rotate_augmented(
    attn: Any,
    raw: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    action_tokens: int,
    *,
    frame_offset: int = 0,
) -> torch.Tensor:
    """Apply axial AC RoPE to raw Q/K in augmented per-frame token order."""
    import torch

    batch, heads, _tokens, head_dim = raw.shape
    per_frame = action_tokens + height * width
    grouped = raw.view(batch, heads, frames, per_frame, head_dim)
    cond_raw = grouped[:, :, :, :action_tokens]
    visual = grouped[:, :, :, action_tokens:].flatten(2, 3)

    frame_pos = torch.arange(
        frame_offset, frame_offset + frames, device=raw.device, dtype=raw.dtype
    )
    cond_parts = []
    for index in range(action_tokens):
        token = cond_raw[:, :, :, index]
        token_d = _ac_bug_compatible_rope(token[..., : attn.d_dim], frame_pos)
        cond_parts.append(torch.cat([token_d, token[..., attn.d_dim :]], dim=-1))
    cond = torch.stack(cond_parts, dim=3).flatten(2, 3)

    ids = torch.arange(
        frame_offset * height * width,
        (frame_offset + frames) * height * width,
        device=raw.device,
    )
    d_pos, h_pos, w_pos = attn.separate_positions(ids, height, width)
    h_pos = h_pos * (attn.grid_size / height)
    w_pos = w_pos * (attn.grid_size / width)
    start = 0
    vd = _ac_bug_compatible_rope(visual[..., start : start + attn.d_dim], d_pos)
    start += attn.d_dim
    vh = _ac_bug_compatible_rope(visual[..., start : start + attn.h_dim], h_pos)
    start += attn.h_dim
    vw = _ac_bug_compatible_rope(visual[..., start : start + attn.w_dim], w_pos)
    start += attn.w_dim
    tail = visual[..., start:]
    visual = torch.cat([vd, vh, vw, tail], dim=-1)

    cond = cond.view(batch, heads, frames, action_tokens, head_dim)
    visual = visual.view(batch, heads, frames, height * width, head_dim)
    return torch.cat([cond, visual], dim=3).flatten(2, 3)


def _sdpa_dtype_harmonizer() -> Any:
    """Scoped patch: cast SDPA's q/k to v's dtype at the call boundary.

    The upstream AC predictor's RoPE attention upcasts q/k to float32 before
    ``F.scaled_dot_product_attention`` while v stays in the weight dtype, which
    is why the predictor has run in fp32 (the June bench identified this as the
    dominant per-forward cost). Under this guard a bf16 predictor computes bf16
    SDPA: q/k are cast back down where they meet v. Process-global while
    active (the adapter scopes it to a single forward; serving is
    single-threaded per engine). GPU-verified 2026-07-11 on H100: energy parity
    with fp32 to bf16 resolution and 4.3x on the batched plan
    (``docs/LEVERS_2026_07_H100.md``); prerequisite for flash-attn on ROCm.
    """
    import contextlib

    import torch.nn.functional as F  # noqa: N812

    @contextlib.contextmanager
    def _guard() -> Any:
        orig = F.scaled_dot_product_attention

        def harmonized(q: Any, k: Any, v: Any, *args: Any, **kwargs: Any) -> Any:
            if q.dtype != v.dtype:
                q = q.to(v.dtype)
            if k.dtype != v.dtype:
                k = k.to(v.dtype)
            return orig(q, k, v, *args, **kwargs)

        F.scaled_dot_product_attention = harmonized  # type: ignore[assignment]
        try:
            yield
        finally:
            F.scaled_dot_product_attention = orig

    return _guard()


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``b`` is exactly the context a cache entry was last synced to.

    Value equality, not identity — a reconstructed ``WorldState`` with
    identical content but a different tensor object should still count as
    "in sync" (see :class:`_CacheEntry`). Cheap relative to a predictor
    forward; a shape check short-circuits the common branch-mismatch case
    without the ``torch.equal`` pass.
    """
    import torch

    return a.shape == b.shape and bool(torch.equal(a, b))


def _infer_tokens_per_frame(encoder: Any) -> int:
    """Spatial patch tokens per temporal frame: ``(crop_size / patch_size) ** 2``.

    Read from the encoder's HF ``config``; falls back to 1 when there is no config
    (the model-agnostic stub path), keeping ``WorldState.context`` rows == frames.
    """
    cfg = getattr(encoder, "config", None)
    crop = getattr(cfg, "crop_size", None)
    patch = getattr(cfg, "patch_size", None)
    if not crop or not patch:
        return 1
    return int((crop // patch) ** 2)


def _decode_observation(
    conditioning: ConditioningInput, *, frames: int, resolution: int
) -> torch.Tensor:
    """Decode an image/video ``uri`` into a ``(1, T, C, H, W)`` float clip.

    A single image is tiled to ``frames``; a video is decoded and sampled to
    ``frames``. Pixels are resized to ``resolution`` and ImageNet-normalized — the
    V-JEPA 2 recipe. Parity with the HF ``VJEPA2VideoProcessor`` is to be confirmed
    on real inputs (Phase 2 serving IO); raises if no ``uri`` is given.
    """
    import torch
    from torchvision.io import read_image, read_video
    from torchvision.transforms import functional as tvf

    uri = conditioning.uri
    if not uri:
        raise ValueError(f"{conditioning.kind} conditioning requires a uri")

    if conditioning.kind is ConditioningKind.IMAGE:
        img = read_image(uri).float() / 255.0  # (C, H, W)
        clip = img.unsqueeze(0).expand(frames, -1, -1, -1)  # (T, C, H, W)
    else:  # VIDEO
        video, _audio, _info = read_video(uri, output_format="TCHW")  # (T, C, H, W)
        video = video.float() / 255.0
        idx = torch.linspace(0, max(video.shape[0] - 1, 0), frames).long()
        clip = video[idx]

    clip = tvf.resize(clip, [resolution, resolution], antialias=True)
    clip = tvf.normalize(clip, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result: torch.Tensor = clip.unsqueeze(0)  # (1, T, C, H, W)
    return result


def _as_context(features: Any) -> torch.Tensor:
    """Normalize an encoder output to a ``(T, D)`` state-embedding context.

    ``get_vision_features`` returns either a tensor or an object carrying
    ``last_hidden_state`` (``scripts/run_vjepa2.py`` handles both); a leading
    batch axis is squeezed.
    """
    tensor = features if hasattr(features, "shape") else features.last_hidden_state
    context: torch.Tensor = tensor[0] if tensor.ndim == 3 else tensor
    return context


def _new_session_id() -> str:
    return uuid.uuid4().hex
