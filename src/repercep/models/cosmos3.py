"""Cosmos 3 Nano engine — the fourth model on the interactive seam.

Cosmos 3 (NVIDIA, released 2026-05-31) is a 16B **omnimodal world model** on a
Mixture-of-Transformers architecture: an autoregressive transformer for text
reasoning and a diffusion transformer for continuous multimodal synthesis,
sharing 3D rotary position embeddings. The ``Cosmos3-Nano-Policy-DROID``
checkpoint this module targets is the post-trained *policy* variant — given a
conditioning frame and a task instruction it jointly predicts the next 17
frames and a 16-step end-effector action chunk. Same policy regime as
LingBot-VA and DreamZero (the model is its own planner, no external CEM), same
DROID robot as DreamZero, but a **different action representation** —
end-effector pose deltas rather than joint positions. See
``docs/COSMOS3_PORT_PLAN.md`` for the full scoping; every geometry constant
below is verified against ``NVIDIA/cosmos``'s
``cookbooks/cosmos3/generator/action/`` notebooks, not the press release.

**The structural difference that shapes this whole module (port plan §2.1):
Cosmos 3 is stateless per chunk.** The reference's "autoregressive rollout"
(``run_fd_with_diffusers.ipynb``'s ``run_rollout``) chains chunks by feeding
each call's *last decoded RGB frame* into the next as the conditioning image;
nothing else crosses the boundary. There is no KV cache, no session-keyed
engine state, and no window bookkeeping — so unlike ``lingbot_va.py`` and
``dreamzero.py`` this engine has no ``_Session`` dict and no ``release()``.

Two consequences worth stating rather than leaving implicit:

* **The seam's branching guarantee actually holds here.**
  ``InteractiveWorldModel.step`` promises the previous state is not mutated so
  a planner can branch from a shared prefix. The two prior interactive ports
  had to break that promise — their native caches mutate in place. A Cosmos 3
  ``WorldState`` is a conditioning frame plus a step counter, so branching is
  free and correct. First port where the seam works as designed.
* **Long rollouts decay.** Chaining on generated pixels compounds
  generation error with no latent path to keep it honest. Real closed-loop
  use re-conditions on a *real* observation each step, which resets the drift;
  imagination-mode rollouts do not, and their quality bounds how many chunks
  a benchmark can honestly report.

The two seam verbs map onto the checkpoint's two action modes rather than onto
cache operations:

* :meth:`Cosmos3Engine.plan` → ``mode="policy"``: the model proposes.
* :meth:`Cosmos3Engine.step` → ``mode="forward_dynamics"``: the caller's
  *executed* chunk drives the prediction, which is what makes ``step``
  action-conditioned at all.

**Unverified, and flagged rather than assumed:** the forward-dynamics path is
demonstrated on the base ``Cosmos3-Nano`` checkpoint, while policy is
demonstrated on ``Cosmos3-Nano-Policy-DROID``. Whether the post-trained policy
checkpoint still serves ``forward_dynamics`` is a Phase-1 gate
(``Cosmos3Config.step_mode`` exists to switch it). If it does not, ``step``
falls back to policy mode and the executed action stops influencing the
rollout — a real semantic loss that must be recorded in the bench doc, not
papered over.

What is **implemented and tested** here is the model-agnostic chunk loop:
conditioning-canvas construction, the step/plan dispatch, action
denormalization and validation. They run against an injected ``pipeline``
(see :class:`_Cosmos3Pipeline`), so the loop is exercised on CPU without
weights — the ``vjepa2_ac.py`` / ``lingbot_va.py`` / ``dreamzero.py`` pattern.
The model-specific pipeline (Phase 1, not landed) wraps diffusers'
``Cosmos3OmniPipeline``; :meth:`Cosmos3Engine.load` raises with the recipe.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from repercep.runtime.engine import EngineInfo
from repercep.runtime.types import Action, LatentStep, WorldState

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend
    from repercep.runtime.types import ConditioningInput, RolloutParams

#: The post-trained DROID policy checkpoint (HF, OpenMDW-1.1). The base
#: ``nvidia/Cosmos3-Nano`` serves forward/inverse dynamics but not policy;
#: ``nvidia/Cosmos3-Edge-Policy-DROID`` is the 4B edge sibling.
DEFAULT_REPO = "nvidia/Cosmos3-Nano-Policy-DROID"

#: The Cosmos safety guardrail asset repository — **gated**, requires an
#: access request on HF. Shared with the Cosmos-Predict1 engine
#: (``models/cosmos.py``), so the account may already be approved.
GUARDRAIL_REPO = "nvidia/Cosmos-1.0-Guardrail"

#: Action-mode tags accepted by ``CosmosActionCondition.mode``.
POLICY_MODE = "policy"
FORWARD_DYNAMICS_MODE = "forward_dynamics"


class _Cosmos3Pipeline(Protocol):
    """The model-specific half of the port, as an injectable pipeline.

    Deliberately a *single* stateless call, because that is what the model is
    (port plan §2.1) — there is no reset/close pair to mirror, no session id to
    key a cache by, and inventing one would misrepresent the model. Wraps

        ``Cosmos3OmniPipeline(prompt=..., action=CosmosActionCondition(...), ...)``

    returning that call's ``(result.video, result.action[0])``. Keeping it a
    Protocol isolates the chunk loop from the port, so the loop is testable on
    CPU with fakes.
    """

    def resolve_conditioning(self, conditioning: ConditioningInput) -> torch.Tensor:
        """Resolve a wire ``ConditioningInput`` to the model's input canvas.

        Owned by the pipeline rather than the engine because it is entirely
        model-specific: it resolves the URI, decodes the three DROID camera
        views, and composites them to the reference geometry (the engine
        exposes :meth:`Cosmos3Engine.build_conditioning_canvas` for the
        compositing half, which is pure PIL and CPU-testable).
        """
        ...

    def infer_chunk(
        self,
        *,
        conditioning: torch.Tensor,
        prompt: str,
        mode: str,
        raw_actions: torch.Tensor | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one pipeline call.

        Returns ``(frames (F, H, W, 3), actions_norm (K, action_dim) | None)``.
        ``actions_norm`` is in the model's normalized space — denormalization
        is the engine's job (:meth:`Cosmos3Engine._denormalize`) — and is
        ``None`` in forward-dynamics mode, where the caller supplied the
        actions and the model returns only frames.
        """
        ...


@dataclass(slots=True)
class Cosmos3Config:
    """Load-time + rollout configuration for :class:`Cosmos3Engine`.

    Defaults are the DROID policy values read from
    ``run_policy_with_diffusers.ipynb``'s ``ACTION_SETS["droid_policy"]`` and
    ``FIXED_SAMPLING`` (verified 2026-08-08), **not** the forward-dynamics
    notebook's — the two differ in ``flow_shift`` (policy 5.0, fd 10.0) and
    the cookbook README documents only the fd value, which is the more
    discoverable of the two and would send a reimplementation silently onto
    the wrong sampling trajectory.
    """

    repo: str = DEFAULT_REPO
    device_index: int = 0
    # BF16 only — the model card is explicit that FP4/FP8/FP16 are untested
    # and unsupported, which removes the usual quantization rungs from the
    # levers ladder before Phase 2 starts.
    dtype: str = "bfloat16"
    # The task instruction; Cosmos 3's goal conditioning is textual.
    prompt: str = "Pick up the object and place it in the target container."

    # --- action-condition geometry (CosmosActionCondition fields) ---
    #: Note the ``_lerobot`` suffix — plain ``"droid"`` is not a valid domain.
    domain_name: str = "droid_lerobot"
    #: Frames generated per call is ``chunk_size + 1``; the pipeline derives it,
    #: which is why height/width/num_frames stay unset on the call.
    chunk_size: int = 16
    resolution_tier: int = 480
    view_point: str = "concat_view"
    fps: int = 15
    #: Which mode ``step()`` uses. ``forward_dynamics`` makes ``step``
    #: genuinely action-conditioned; falling back to ``policy`` means the
    #: executed action is ignored (see the module docstring's Phase-1 gate).
    step_mode: str = FORWARD_DYNAMICS_MODE

    # --- sampling ---
    num_inference_steps: int = 30
    #: 1.0 = no classifier-free guidance. Worth noticing as a *lever that is
    #: absent*: DreamZero's CFG-batching win (cond+uncond in one forward) has
    #: no analogue here because there is no unconditional pass to batch.
    guidance_scale: float = 1.0
    #: ``UniPCMultistepScheduler(flow_shift=..., use_karras_sigmas=False)``.
    flow_shift: float = 5.0
    seed: int = 0

    # --- DROID action layout (cookbook README's action table) ---
    #: 10D = end-effector pose delta (9D: 3D translation + 6D continuous
    #: rotation) + gripper grasp state (1D), in meters. Contrast DreamZero-DROID
    #: on the *same robot*: 8D joint-space. Same embodiment, different action
    #: representation — a side-by-side table must say so inline.
    action_dim: int = 10
    #: Channels ``[0:9]`` are the ``rot6d`` pose delta consumed by
    #: ``pose_rel_to_abs(..., pose_convention="backward_framewise")``;
    #: channel 9 is the gripper.
    pose_dim: int = 9

    # --- conditioning canvas (build_concat_frame) ---
    #: The three DROID cameras are composited into ONE image before the
    #: pipeline sees them: wrist across the full top half, exterior_1 and
    #: exterior_2 side by side across the bottom half. This is notebook
    #: pre-processing, not something the pipeline does, so the engine owns it —
    #: and its geometry is part of the model contract.
    canvas_width: int = 640
    canvas_height: int = 540

    # --- deployment ---
    #: The Generator requires the guardrail unless explicitly disabled.
    enable_guardrail: bool = False
    #: ``Cosmos3OmniPipeline``/``CosmosActionCondition`` exist only on diffusers
    #: ``main`` — no released version carries them. Phase 1 must pin an exact
    #: commit here and record it in the bench doc: a floating ``main`` under a
    #: leaderboard row is how reproducibility dies quietly.
    diffusers_commit: str | None = None


class Cosmos3Engine:
    """Cosmos 3 Nano served on a Repercep backend (the interactive seam).

    Thin by construction — the model carries no cross-chunk state (module
    docstring), so this engine holds only a pipeline and a config. There is no
    per-session bookkeeping to leak, which is why, unlike its two predecessors,
    it exposes no ``release()``.

    ``WorldState.context`` carries the conditioning canvas ``(H, W, 3)`` for
    the next call — the model's entire memory of the rollout.
    """

    model_name = "cosmos3-nano-policy-droid"

    def __init__(
        self,
        backend: Backend,
        config: Cosmos3Config | None = None,
        *,
        pipeline: _Cosmos3Pipeline | None = None,
        action_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self._backend = backend
        self._config = config if config is not None else Cosmos3Config()
        # Injected for testing / advanced use; otherwise built by ``load()``
        # once Phase 1 lands.
        self._pipeline: _Cosmos3Pipeline | None = pipeline
        # ``(q01, q99)`` per-channel DROID quantiles. Injectable for the same
        # reason ``pipeline`` is — otherwise ``plan()`` is untestable without
        # weights, since it refuses to return normalized values (see
        # :meth:`_denormalize`). Phase 1's ``load()`` populates it from
        # ``droid_lerobot_stats.json``.
        self._action_stats = action_stats

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
        """Build the real diffusers pipeline. **Phase 1, not implemented.**

        The recipe, so it is not re-derived from scratch later
        (``docs/COSMOS3_PORT_PLAN.md`` §2):

        1. Own venv. ``diffusers`` from **git main at a pinned SHA** — the
           released wheels have no ``Cosmos3OmniPipeline``. Set
           :attr:`Cosmos3Config.diffusers_commit` and record it.
        2. ``Cosmos3OmniPipeline.from_pretrained(repo, torch_dtype=bfloat16,
           safety_checker=None, enable_safety_checker=<config>)``, then
           ``.to(device)``; replace the scheduler with
           ``UniPCMultistepScheduler.from_config(pipe.scheduler.config,
           flow_shift=5.0, use_karras_sigmas=False)``.
        3. Guardrail access (``nvidia/Cosmos-1.0-Guardrail``) is **gated** —
           request it before booking GPU time; the Cosmos-Predict1 engine
           already uses this repo, so the account may be approved already.
        4. Do **not** port the Cosmos Framework path
           (``action_policy_server_robolab`` + the RoboLab client): that is an
           eval harness, and the standing lesson from LingBot-VA's ``VA_Server``
           and DreamZero's ``ARDroidRoboarenaPolicy`` is to port the component
           and never the harness above it.
        5. Load DROID quantile stats from the framework checkout's
           ``normalizer_stats/droid_lerobot_stats.json`` into
           :attr:`_action_stats` (see :meth:`_denormalize`).
        """
        raise NotImplementedError(
            "Cosmos 3 Nano weight loading is Phase 1 (GPU) — see this method's "
            "docstring and docs/COSMOS3_PORT_PLAN.md §2/§4 for the port recipe. "
            "Inject a pipeline to exercise the chunk loop on CPU."
        )

    # --- the interactive seam ---

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Seed a rollout from an observation.

        No pipeline call and no server-side allocation — for a stateless model
        a reset is just "here is the first conditioning frame." The caller is
        expected to have composited the three DROID camera views already (see
        :meth:`build_conditioning_canvas`, which does it to the reference
        geometry); ``conditioning.uri`` resolution is the serving layer's job,
        as on every other engine.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        canvas = self._pipeline.resolve_conditioning(conditioning)
        return WorldState(context=canvas, step_index=0, session_id=uuid.uuid4().hex)

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one chunk under the *executed* action chunk.

        Runs :attr:`Cosmos3Config.step_mode` (forward dynamics by default) with
        ``action.values`` as the driving trajectory, and chains: the new
        state's conditioning frame is this call's **last generated frame**,
        matching the reference ``run_rollout`` exactly, including its discard
        of frame 0 (each call regenerates its own conditioning frame as frame
        0, so keeping it would duplicate a frame per chunk).

        ``action.values`` is a flat executed chunk — ``k x action_dim``
        floats, ``space="cosmos3_chunk_droid_ee"``. The ``_ee`` is load-bearing:
        these are end-effector pose deltas, not the joint-space values a
        DreamZero ``Action`` carries for the same robot.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        actions = self._parse_action_chunk(action, like=state.context)
        mode = self._config.step_mode
        frames, _ = self._pipeline.infer_chunk(
            conditioning=state.context,
            prompt=self._config.prompt,
            mode=mode,
            raw_actions=actions if mode == FORWARD_DYNAMICS_MODE else None,
            # Vary the seed per chunk, as the reference rollout does
            # (``seed=chunk_index``) — a fixed seed across chunks correlates
            # the noise draws and is not what the reference measures.
            seed=self._config.seed + state.step_index,
        )
        return (
            WorldState(
                context=self._last_frame(frames),
                step_index=state.step_index + 1,
                session_id=state.session_id,
            ),
            LatentStep(step_index=state.step_index + 1),
        )

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Policy-mode planning: the model's own proposed action chunk.

        Cosmos 3 *generates* actions (the policy head is the planner), so
        ``goal``/``horizon`` are accepted for seam compatibility and unused —
        goal conditioning is the text instruction in
        :attr:`Cosmos3Config.prompt`. Returns the first action of the proposed
        chunk, denormalized.

        Unlike the two prior interactive ports there is no parked-proposal
        cache: a policy call here is stateless, so ``plan`` simply makes one.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        _, proposed = self._pipeline.infer_chunk(
            conditioning=state.context,
            prompt=self._config.prompt,
            mode=POLICY_MODE,
            raw_actions=None,
            seed=self._config.seed + state.step_index,
        )
        if proposed is None:
            raise RuntimeError(
                f"pipeline returned no actions in {POLICY_MODE!r} mode — a policy "
                "call must produce an action chunk"
            )
        values = self._denormalize(proposed)[0]
        return Action(values=values.tolist(), space="cosmos3_chunk_droid_ee")

    # --- conditioning canvas ---

    def build_conditioning_canvas(
        self,
        *,
        wrist: torch.Tensor,
        exterior_1: torch.Tensor,
        exterior_2: torch.Tensor,
    ) -> torch.Tensor:
        """Composite the three DROID camera views into the model's input canvas.

        Reproduces the reference ``build_concat_frame``: a
        ``canvas_width x canvas_height`` (640x540) RGB canvas with ``wrist``
        across the full top half and ``exterior_1`` / ``exterior_2`` side by
        side across the bottom half, each fitted with ``ImageOps.fit`` —
        centre-crop to the tile's aspect ratio, then bicubic resize.

        PIL does the fitting rather than a torch reimplementation on purpose:
        ``ImageOps.fit``'s crop-then-bicubic is what produced the conditioning
        the checkpoint was tuned on, and torch's bicubic differs enough at the
        edges to put an exact-parity gate at risk for no benefit.

        Inputs and output are ``(H, W, 3)`` uint8 tensors.
        """
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        width, height = self._config.canvas_width, self._config.canvas_height
        top_height = height // 2
        bottom_height = height - top_height
        half_width = width // 2

        canvas = Image.new("RGB", (width, height))
        tiles = (
            (wrist, (width, top_height), (0, 0)),
            (exterior_1, (half_width, bottom_height), (0, top_height)),
            (exterior_2, (half_width, bottom_height), (half_width, top_height)),
        )
        for frame, size, position in tiles:
            image = Image.fromarray(frame.detach().cpu().numpy().astype(np.uint8), mode="RGB")
            canvas.paste(ImageOps.fit(image, size, method=Image.Resampling.BICUBIC), position)
        return torch.from_numpy(np.asarray(canvas).copy())

    # --- internals ---

    def _require_pipeline(self) -> None:
        if self._pipeline is None:
            self.load()

    def _last_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """The chained conditioning frame for the next chunk.

        The reference keeps ``chunk_frames[1:]`` for output and chains on
        ``chunk_frames[-1]``; both indices come off the same tensor here.
        """
        if frames.shape[0] < 1:
            raise ValueError("pipeline returned no frames")
        return frames[-1]

    def _denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Invert the checkpoint's quantile normalization.

        Stats come from the Cosmos Framework's
        ``normalizer_stats/droid_lerobot_stats.json`` and are loaded by
        :meth:`load` (Phase 1). Without them the honest thing is to raise:
        returning normalized values that *look* like meters would produce a
        plausible, entirely wrong trajectory — the failure mode
        ``action_norm.denormalize_quantile``'s ``eps`` note also warns about.
        """
        from repercep.models.action_norm import denormalize_quantile

        if self._action_stats is None:
            raise RuntimeError(
                "DROID action quantile stats not loaded — Phase 1 loads "
                "droid_lerobot_stats.json in load(); refusing to return "
                "normalized values as if they were meters"
            )
        q01, q99 = self._action_stats
        return denormalize_quantile(actions, q01, q99, eps=0.0)

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
