"""Model-specific Phase-1 GPU port for the VLA action-token engine.

Builds the real :class:`~repercep.models.vla._VLAPipeline` from an OpenVLA /
π0-FAST bundle: the processor + action detokenizer and the KV-cached
autoregressive action-token decode over the VLM backbone. Until that lands,
:func:`build_pipeline` raises with the setup recipe — the model-agnostic action
loop in :mod:`repercep.models.vla` is exercised on CPU with an injected fake
(see ``docs/VLA_PORT_PLAN.md``, the ``lingbot_va`` pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from repercep.backend.protocol import Backend
    from repercep.models.vla import VLAConfig, _VLAPipeline


def build_pipeline(backend: Backend, config: VLAConfig) -> _VLAPipeline:
    """Construct the real VLA pipeline. Phase-1 GPU port — raises until landed."""
    raise NotImplementedError(
        "VLA action-token decode is a Phase-1 GPU port (see docs/VLA_PORT_PLAN.md). "
        f"It needs the model bundle ({config.repo}) + transformers, a GPU, and the "
        "action detokenizer. Inject a _VLAPipeline (the fake-pipeline pattern) for "
        "CPU development and testing instead."
    )
