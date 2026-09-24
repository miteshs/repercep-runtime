"""FlashAttention on H100 via Dao-AILab ``flash-attn``.

The implementation plan named FlashAttention-3 as the attention primitive on
NVIDIA silicon; on Hopper that is exactly what we get.  This module wraps the
Dao-AILab ``flash-attn`` package behind ``AttentionOp`` so the substitution is
invisible above the backend layer (ADR-0003).

Two import paths are tried in order:

1. ``flash_attn_interface.flash_attn_func`` — the FA-3 entry point, shipped
   alongside FA-2 in the same wheel as of ``flash-attn>=2.7``.  Hopper-only
   (uses ``wgmma`` + TMA via cuDNN).  This is the preferred path on H100/H200.
2. ``flash_attn.flash_attn_func`` — the FA-2 entry point.  Falls back to this
   on Ampere/Ada where FA-3 is unavailable, and as a defensive backstop on
   Hopper if FA-3 is not built into the wheel for some reason.

If neither is importable the op advertises ``available == False`` and selection
in :mod:`repercep.attention.registry` falls through to :class:`NaiveAttention`.

ROCm sibling: :class:`repercep.attention.rocm_flash.ROCmFlashAttention` wraps the
AMD Composable-Kernel ``flash-attn`` build for gfx942; the two are deliberate
parallels, never one parameterized wrapper — they target different physical
kernels with different layout + dtype rules (see ADR-0002 + ADR-0006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


# head_dim values both FA-3 and FA-2 ship compiled tiles for on Hopper.
# FA-3 additionally supports 64/96/128/192/256; we take the intersection so a
# fallback from FA-3 to FA-2 doesn't surprise on an unusual head_dim.
_SUPPORTED_HEAD_DIMS = frozenset({64, 96, 128, 192, 256})

# FA-3 supports FP16 / BF16 / FP8.  The FP8 path here is the BF16 entry point
# (the kernel takes BF16/FP16 inputs and quantizes internally).  Pure-FP8
# inputs go through :class:`TransformerEngineAttention` or
# :class:`FP8HopperTritonAttention` — we keep this wrapper as the BF16/FP16
# flash path so the dispatch tree mirrors ROCm's exactly.
_SUPPORTED_DTYPES = frozenset({DType.FP16, DType.BF16})

# Kinds: FULL and CAUSAL.  Neighborhood attention is a future kernel (a
# Hopper port of NATTEN's CUDA build would land here when needed).
_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


class HopperFlashAttention:
    """FA-3-preferred, FA-2-fallback flash attention for Hopper / Ada / Ampere."""

    name = "nvidia-flash"

    def __init__(self) -> None:
        self._fn: Any | None = None
        self._is_fa3: bool = False
        self._import_error: str | None = None
        # Try FA-3 first (Hopper-only entry point).
        try:
            from flash_attn_interface import flash_attn_func as _fa3

            self._fn = _fa3
            self._is_fa3 = True
            return
        except ImportError as exc:
            self._import_error = f"flash_attn_interface (FA-3): {exc}"

        # Fall back to FA-2 (Ampere/Ada/Hopper).
        try:
            from flash_attn import flash_attn_func as _fa2

            self._fn = _fa2
            self._is_fa3 = False
            return
        except ImportError as exc:
            self._import_error = f"{self._import_error}; flash_attn (FA-2): {exc}"
            self._fn = None

    @property
    def available(self) -> bool:
        """True if either FA-3 or FA-2 is importable."""
        return self._fn is not None

    @property
    def is_fa3(self) -> bool:
        """True if the FA-3 entry point was the one resolved."""
        return self._is_fa3

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
                "HopperFlashAttention requires flash-attn. "
                f"Last import error: {self._import_error}. "
                "Install via `pip install flash-attn --no-build-isolation` "
                "(see docs/COSMOS_ON_H100.md §Reproduce) or "
                "pin attention_backend='naive-sdpa'."
            )
        # Repercep convention is (B, H, S, D); flash_attn_func wants (B, S, H, D).
        # Both FA-3 and FA-2 share this convention; we transpose in and back.
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out_or_pair = self._fn(q, k, v, causal=causal, softmax_scale=scale)
        # FA-3 returns (out, lse); FA-2 returns out alone.  Accept both.
        out = out_or_pair[0] if isinstance(out_or_pair, tuple) else out_or_pair
        return out.transpose(1, 2)  # type: ignore[no-any-return]
