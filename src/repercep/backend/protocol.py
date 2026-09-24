"""The ``Backend`` Protocol — the contract every compute backend implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from repercep.attention.protocol import AttentionOp
    from repercep.attention.types import AttentionShape
    from repercep.hardware import BackendCapabilities, DeviceSpec, DType, Vendor


@runtime_checkable
class Backend(Protocol):
    """A compute substrate Repercep can run a world model on.

    This is the single seam between Repercep and a GPU vendor.  Everything above
    it — the Runtime, the model code, the serving layer — is written against
    this Protocol only.  Adding NVIDIA support means writing one more class
    that satisfies this Protocol; no existing code changes.  That is the whole
    point of leading with MI300X without painting the project into a corner
    (ADR-0001).
    """

    vendor: Vendor
    name: str

    def is_available(self) -> bool:
        """True if this backend's hardware is present and usable on this host."""
        ...

    def devices(self) -> tuple[DeviceSpec, ...]:
        """Every accelerator this backend can address, in index order."""
        ...

    def capabilities(self) -> BackendCapabilities:
        """Dtype support, attention ops, and feature flags for this host."""
        ...

    def torch_device(self, index: int = 0) -> torch.device:
        """The ``torch.device`` handle for the accelerator at ``index``."""
        ...

    def default_dtype(self) -> DType:
        """The preferred compute dtype for this architecture."""
        ...

    def attention_op(self, shape: AttentionShape, dtype: DType) -> AttentionOp:
        """The fastest attention implementation for the given problem shape."""
        ...
