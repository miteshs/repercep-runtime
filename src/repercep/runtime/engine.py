"""The Runtime's top-level seam: the ``WorldModelEngine`` Protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Iterator

    from repercep.runtime.types import Frame, GenerationRequest


class EngineInfo(BaseModel):
    """A snapshot of an engine's identity and readiness."""

    # `model_name` collides with Pydantic's reserved `model_` namespace; the
    # field name is worth keeping, so the namespace guard is disabled here.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    backend: str
    device: str
    dtype: str
    ready: bool


@runtime_checkable
class WorldModelEngine(Protocol):
    """Turns a ``GenerationRequest`` into a stream of frames.

    This is the seam the serving layer depends on. The Cosmos-Predict-7B engine
    (Task #7) and the ``StubEngine`` used to build the serving stack today are
    interchangeable because both satisfy this Protocol.
    """

    def info(self) -> EngineInfo:
        """Identity and readiness of this engine."""
        ...

    def generate(self, request: GenerationRequest) -> Iterator[Frame]:
        """Yield frames in order, lazily.

        Lazy iteration is what makes frame-level streaming work: a slow
        consumer backpressures the producer instead of forcing the whole clip
        to be materialized up front.
        """
        ...
