"""A no-model engine for building and testing the serving stack.

``StubEngine`` satisfies ``WorldModelEngine`` and yields deterministic noise
frames. It exists so the HTTP/gRPC layer, streaming, and latent-cache wiring
can be developed and tested before the Cosmos-Predict-7B loader lands (Task
#7). It is explicitly not a model — every frame is random noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from repercep.runtime.engine import EngineInfo
from repercep.runtime.types import Frame

if TYPE_CHECKING:
    from collections.abc import Iterator

    from repercep.runtime.types import GenerationRequest


class StubEngine:
    """Yields random-noise frames sized to the request."""

    model_name = "stub-noise"

    def __init__(self, backend_name: str = "none", device: str = "cpu") -> None:
        self._backend = backend_name
        self._device = device

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name=self.model_name,
            backend=self._backend,
            device=self._device,
            dtype="uint8",
            ready=True,
        )

    def generate(self, request: GenerationRequest) -> Iterator[Frame]:
        import torch

        params = request.params
        generator = torch.Generator()
        if params.seed is not None:
            generator.manual_seed(params.seed)
        for index in range(params.num_frames):
            pixels = torch.randint(
                0,
                256,
                (params.height, params.width, 3),
                generator=generator,
                dtype=torch.uint8,
            )
            yield Frame(index=index, total=params.num_frames, pixels=pixels)
