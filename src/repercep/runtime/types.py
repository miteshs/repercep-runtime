"""Request, response, and frame types for the Repercep Runtime.

These types are the wire contract. Every externally-visible type is a Pydantic
model, so the HTTP and gRPC layers serialize it directly and a malformed
request fails validation at the edge rather than deep in the diffusion loop.
``Frame`` is the one internal exception — it carries a live tensor and never
crosses the wire.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import torch

_STRICT = ConfigDict(extra="forbid")


class ConditioningKind(enum.StrEnum):
    """How a generation is conditioned on visual input."""

    NONE = "none"  # pure Text2World
    IMAGE = "image"  # Video2World rolled out from a single frame
    VIDEO = "video"  # Video2World continuing an existing clip


class ConditioningInput(BaseModel):
    """Optional visual conditioning for Video2World generation.

    Raw pixels never travel in JSON. ``uri`` is a reference the serving layer
    resolves: a data URI, an object-store key, or a server-local path.
    """

    model_config = _STRICT

    kind: ConditioningKind = ConditioningKind.NONE
    uri: str | None = None


class GenerationParams(BaseModel):
    """Diffusion and sampling knobs.

    Defaults track the Cosmos-Predict-7B reference configuration (121 frames at
    704x1280, 35 denoising steps). Bounds are deliberately conservative for
    v0.1 and will widen as the runtime is hardened.
    """

    model_config = _STRICT

    num_frames: int = Field(121, ge=1, le=256)
    height: int = Field(704, ge=64, le=2048)
    width: int = Field(1280, ge=64, le=2048)
    num_inference_steps: int = Field(35, ge=1, le=200)
    guidance_scale: float = Field(7.0, ge=0.0, le=30.0)
    fps: int = Field(24, ge=1, le=120)
    seed: int | None = Field(None, description="None yields a nondeterministic run.")


class GenerationRequest(BaseModel):
    """A single world-model generation request."""

    model_config = _STRICT

    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    params: GenerationParams = Field(default_factory=GenerationParams)
    conditioning: ConditioningInput = Field(default_factory=ConditioningInput)


@dataclass(slots=True)
class Frame:
    """One generated frame, in-process.

    Internal type: ``pixels`` is a live tensor, so ``Frame`` never crosses the
    wire — the serving layer converts it to a ``FrameChunk`` plus an
    out-of-band pixel payload.
    """

    index: int
    total: int
    pixels: torch.Tensor  # (height, width, 3)


class FrameChunk(BaseModel):
    """One frame as it streams to a client — the metadata envelope.

    The pixel payload is carried out of band (a follow-up binary message, or an
    object-store URI in ``uri``); keeping bytes out of the JSON keeps the
    stream cheap to parse and log.
    """

    model_config = _STRICT

    frame_index: int
    total_frames: int
    height: int
    width: int
    latency_ms: float
    uri: str | None = None


class GenerationResult(BaseModel):
    """Terminal summary of a completed generation."""

    model_config = _STRICT

    num_frames: int
    total_latency_ms: float
    frames_per_second: float
    backend: str
    device: str
    attention_op: str


class Action(BaseModel):
    """One control input to an interactive (action-conditioned) world model.

    Actions live in a latent/continuous control space — a robot end-effector
    delta, a steering command, a discrete game input encoded as a vector —
    never pixels. ``space`` tags the interpretation so an engine can validate
    dimensionality at the edge, the same way every other request type fails
    fast on malformed input.
    """

    model_config = _STRICT

    values: list[float] = Field(..., min_length=1)
    space: str = Field("raw", description="Control-space tag, e.g. 'ee_delta_7d'.")


class RolloutParams(BaseModel):
    """Knobs for an interactive rollout.

    ``horizon`` is the interactive analogue of ``GenerationParams.num_frames``:
    how many latent steps a planning rollout looks ahead. ``decode_pixels`` is
    off by default — a latent world model (V-JEPA 2-AC) has no decoder, so its
    steps carry embeddings, not frames; an action-conditioned video-diffusion
    engine can flip it on.
    """

    model_config = _STRICT

    horizon: int = Field(16, ge=1, le=512)
    decode_pixels: bool = False
    return_energy: bool = True


class ResetRequest(BaseModel):
    """Open or re-seed an interactive world-model session.

    The world is seeded from an observation via the same ``ConditioningInput``
    the one-shot path uses (``kind=image|video`` + ``uri``); ``NONE`` seeds an
    unconditioned rollout from the model's learned prior.
    """

    model_config = _STRICT

    conditioning: ConditioningInput = Field(default_factory=ConditioningInput)
    params: RolloutParams = Field(default_factory=RolloutParams)


@dataclass(slots=True)
class WorldState:
    """The rolling latent context of an interactive world model, in-process.

    Internal type, like :class:`Frame`: ``context`` is a live tensor — the
    recent window of state embeddings the predictor attends over (block-causal)
    — so ``WorldState`` never crosses the wire. The serving layer holds it per
    session and streams only :class:`LatentStep` envelopes back to the client.
    """

    context: torch.Tensor  # (T_ctx, D) recent state embeddings
    step_index: int
    session_id: str


class LatentStep(BaseModel):
    """One interactive step as it streams to a client — the metadata envelope.

    Like :class:`FrameChunk`, no tensor crosses the wire. ``energy`` is the
    latent-space cost of the step relative to a goal — the scalar an
    energy-based planner minimizes; ``frame`` is populated only when the engine
    has a pixel decoder attached and ``decode_pixels`` was requested (the
    AVID-style path), and is ``None`` for a pure latent world model.
    """

    model_config = _STRICT

    step_index: int
    energy: float | None = None
    frame: FrameChunk | None = None
