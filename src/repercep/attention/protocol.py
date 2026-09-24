"""The ``AttentionOp`` Protocol — the contract every attention kernel implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from repercep.attention.types import AttentionShape
    from repercep.hardware import DType


@runtime_checkable
class AttentionOp(Protocol):
    """A single attention implementation bound to a backend.

    Implementations wrap a concrete kernel — CK flash-attention, aotriton SDPA,
    a Triton kernel, or the naive reference.  The Runtime obtains an op via
    ``Backend.attention_op`` and never imports kernels directly, so swapping a
    kernel never touches model code.
    """

    name: str

    @property
    def available(self) -> bool:
        """True when the concrete kernel can be used on this host."""
        ...

    def supports(self, shape: AttentionShape, dtype: DType) -> bool:
        """True if this op can correctly run the given problem shape and dtype."""
        ...

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Compute attention.

        Args:
            query, key, value: ``(batch, heads, seq, head_dim)`` tensors.
            causal: apply a causal mask (temporal autoregression).
            scale: softmax scale; ``None`` means ``1/sqrt(head_dim)``.

        Returns:
            ``(batch, heads, seq_len_q, head_dim)``.
        """
        ...
