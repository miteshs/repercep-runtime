"""FlashAttention on MI300X via AMD Composable Kernel.

The implementation plan names FlashAttention-3 as the attention primitive, but
FA-3's kernels are Hopper-specific (they rely on ``wgmma`` and TMA, which CDNA3
does not have).  The MI300X-equivalent primitive is the Composable-Kernel (CK)
``flash-attn`` ROCm build.  This module wraps it behind ``AttentionOp`` so the
substitution is invisible above the backend layer.  See ADR-0002.

This is a thin, honest wrapper today: if the CK ``flash_attn`` package is not
installed it advertises ``available == False`` and ``supports() == False``, and
the registry falls back to ``NaiveAttention``.  Task #6 fills in layout
handling, varlen support, and FP8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch

# head_dim values the CK flash-attn kernels are compiled for on CDNA3.
_SUPPORTED_HEAD_DIMS = frozenset({64, 128, 256})
_SUPPORTED_DTYPES = frozenset({DType.FP16, DType.BF16})
_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


class ROCmFlashAttention:
    """CK-backed FlashAttention for gfx942."""

    name = "rocm-ck-flash"

    def __init__(self) -> None:
        self._fn: Any | None = None
        try:
            from flash_attn import flash_attn_func

            self._fn = flash_attn_func
        except ImportError:
            self._fn = None

    @property
    def available(self) -> bool:
        """True if the CK ``flash-attn`` build is importable."""
        return self._fn is not None

    def supports(self, shape: AttentionShape, dtype: DType) -> bool:
        return (
            self._fn is not None
            and dtype in _SUPPORTED_DTYPES
            and shape.head_dim in _SUPPORTED_HEAD_DIMS
            and shape.kind in _SUPPORTED_KINDS
        )

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        if self._fn is None:
            raise RuntimeError(
                "ROCm flash-attention is not installed. "
                "Build the CK flash-attn package for gfx942 — see ADR-0002 — "
                "or pin attention_backend='naive-sdpa'."
            )
        # Repercep convention is (batch, heads, seq, head_dim); flash_attn_func
        # wants (batch, seq, heads, head_dim).  Transpose in and back out.
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out: torch.Tensor = self._fn(q, k, v, causal=causal, softmax_scale=scale)
        return out.transpose(1, 2)
