"""Intel Extension for PyTorch flash attention on CPU.

IPEX exposes an AMX-aware fused attention through ``ipex.llm.functional
.varlen_attention`` (LLM path, padding-free) and ``ipex.llm.modules
.IndirectAccessKVCacheAttention`` (KV-cache path).  For Repercep's
world-model DiT blocks we want the simpler ``no-KV-cache`` path: a single
fused call that runs QK^T + softmax + PV in tiled AMX matmuls without
materialising the (B, H, S, S) score matrix.

The fused op is the right CPU sibling to FA-2 / FA-3 — the same memory-
hierarchy story (tiles in L2, KV streaming through L1) just on AMX
tile registers instead of TMA + SMEM.  IPEX 2.6+ ships AMX-tuned kernels
for Sapphire Rapids and Emerald Rapids; on older Xeon the same call falls
back to AVX-512 (still fast, no AMX).

Why this is separate from :class:`AMXSDPAAttention`: SDPA's torch path on
CPU still materialises QK^T as an intermediate, even when oneDNN dispatches
the matmul to AMX tiles — peak HBM-equivalent (peak RAM) is O(S^2) instead
of O(S * head_dim).  At the Cosmos shape (S ≈ 83k) the S^2 buffer is
~55 GiB per head per batch — unworkable.  The IPEX path is the only way
to get the flash-style memory profile on CPU today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


_SUPPORTED_HEAD_DIMS = frozenset({64, 80, 96, 128, 192, 256})
_SUPPORTED_DTYPES = frozenset({DType.FP16, DType.BF16, DType.FP32})
_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


class IPEXFlashAttention:
    """IPEX-fused, AMX-aware attention.  Optional (gated on the IPEX install)."""

    name = "ipex-flash"

    def __init__(self) -> None:
        self._fn: Any | None = None
        self._import_error: str | None = None
        # IPEX's public top-level export of the fused attention has moved
        # between releases; we probe the documented stable surface first
        # (``ipex.llm.functional.varlen_attention``), then a torch-native
        # SDPA-compatible variant when present.
        try:
            import intel_extension_for_pytorch as ipex

            # Newer IPEX (2.5+) ships ``ipex.llm.functional.scaled_dot_product_attention``
            # which is a drop-in SDPA with AMX-aware fused kernel selection.
            llm_fn = getattr(getattr(ipex, "llm", None), "functional", None)
            self._fn = getattr(llm_fn, "scaled_dot_product_attention", None)
            if self._fn is None:
                # Fall back to the older ``ipex.nn.functional.flash_attention`` API.
                self._fn = getattr(getattr(ipex, "nn", None), "functional", None)
                self._fn = getattr(self._fn, "flash_attention", None) if self._fn else None
            if self._fn is None:
                self._import_error = "ipex installed but no fused attention entry point found"
        except ImportError as exc:
            self._import_error = f"intel_extension_for_pytorch: {exc}"

    @property
    def available(self) -> bool:
        return self._fn is not None

    def supports(self, shape: AttentionShape, dtype: DType) -> bool:
        return (
            self.available
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
                f"IPEXFlashAttention requires intel-extension-for-pytorch: "
                f"{self._import_error}.  Install with "
                "`uv pip install intel-extension-for-pytorch` against a matching torch."
            )
        # IPEX's fused entrypoint follows SDPA's signature: BHSD in, BHSD out.
        return self._fn(query, key, value, is_causal=causal, scale=scale)  # type: ignore[no-any-return]
