"""The interactive, action-conditioned engine seam: ``InteractiveWorldModel``.

Where :class:`repercep.runtime.engine.WorldModelEngine` turns one request into one
stream of frames, this Protocol models a *closed loop*: the client is in the
loop — it sends an action, receives the next world state, and decides the next
action from it. That is the regime a request/response diffusion server does not
express, and it is where action conditioning and closed-loop latency live (see
``docs/adr/0008-interactive-world-model-seam.md``).

The latent world model (V-JEPA 2-AC, :mod:`repercep.models.vjepa2_ac`) is the lead
implementation: it predicts the next *state embedding* conditioned on an action
and plans by minimizing a latent-space energy over candidate action sequences —
i.e. the energy-based / JEPA world model, served. An action-conditioned
video-diffusion model (AVID-style) can implement the same Protocol while also
decoding pixels per step, reusing the native denoise loop and the adaptive cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from repercep.runtime.engine import EngineInfo
    from repercep.runtime.types import (
        Action,
        ConditioningInput,
        LatentStep,
        RolloutParams,
        WorldState,
    )


@runtime_checkable
class InteractiveWorldModel(Protocol):
    """A stateful, action-conditioned world model — the closed-loop seam.

    Implementations advance a latent :class:`~repercep.runtime.types.WorldState`
    one action at a time. They reuse the same ``Backend`` seam (ADR-0003) and
    config conventions as :class:`~repercep.runtime.engine.WorldModelEngine`; the
    difference is that state persists across calls and the caller drives the
    action sequence.
    """

    def info(self) -> EngineInfo:
        """Identity and readiness of this engine."""
        ...

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Seed a fresh world state from an observation.

        ``conditioning`` reuses the one-shot path's image/video reference; an
        unconditioned (``NONE``) reset starts from the model's learned prior.
        """
        ...

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one latent step under ``action``.

        Returns the new state and the streamable envelope. The previous
        ``state`` is not mutated, so a planner can branch rollouts from a shared
        prefix.
        """
        ...

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Return the next action that minimizes a latent energy toward ``goal``.

        The energy is the embedding-space distance between a rollout's predicted
        terminal state and ``goal``; planning is energy minimization over
        candidate action sequences (e.g. CEM / MPC). Optional — an engine that
        only does open-loop rollout may raise ``NotImplementedError``.
        """
        ...
