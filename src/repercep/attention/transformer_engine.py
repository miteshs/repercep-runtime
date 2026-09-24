"""NVIDIA-canonical FP8 attention via TransformerEngine on Hopper.

TransformerEngine (TE) is NVIDIA's reference path for FP8 attention on H100/H200.
Under the hood it dispatches to the FlashAttention-3 algorithm — Hopper's TMA
(tensor-memory-accelerator) async copies plus WGMMA warpgroup matmul, exposed
through cuDNN's fused flash-attention kernels — and applies an FP8 *recipe* to
manage per-tensor scales across the Q/K/V/output projections.

This op is **optional**. TE has a heavy install footprint: it requires the
matching CUDA toolchain (nvcc) at install time and ships C++ extensions per
SM target. Repercep's lead hardware is MI300X (ADR-0001), and the AMD path uses
``ROCmFlashAttention`` (CK flash-attn) plus ``FP8TritonAttention`` instead
(ADR-0002). TE is registered only when ``transformer_engine`` imports
successfully *and* CUDA is available; otherwise the registry falls back to a
ROCm/Triton op or the naive reference. See ADR-0002 and ADR-0006.

**Recipe note:** TE's FP8 path is driven by a recipe object (``DelayedScaling``
by default) which tracks per-tensor amax history to choose scaling factors.
This wrapper uses the default recipe — adequate for inference at the shapes
Repercep targets. Production users training or serving at extreme aspect ratios
would tune the recipe (history length, margin, scaling format) explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


# TE's FP8 flash-attention path on Hopper is compiled for head_dim 64 and 128.
# Other head dims fall back to non-FP8 kernels inside TE; we keep the supported
# set tight so the registry only picks TE when it can actually deliver FP8.
_SUPPORTED_HEAD_DIMS = frozenset({64, 128})
_SUPPORTED_DTYPES = frozenset({DType.FP16, DType.BF16, DType.FP8_E4M3, DType.FP8_E5M2})
_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


class TransformerEngineAttention:
    """TE-backed FP8 attention for Hopper (H100/H200)."""

    name = "transformer-engine-fp8"

    def __init__(self) -> None:
        self._dpa_cls: Any | None = None
        self._import_error: str | None = None
        self._cache: dict[tuple[int, int, str], Any] = {}
        try:
            import torch as _torch
            from transformer_engine.pytorch import DotProductAttention

            if not _torch.cuda.is_available():
                self._import_error = "CUDA not available"
                self._dpa_cls = None
            else:
                self._dpa_cls = DotProductAttention
        except ImportError as e:  # pragma: no cover - environment dependent
            self._import_error = str(e)
            self._dpa_cls = None
        except Exception as e:  # pragma: no cover - TE init can fail at import
            # TE sometimes raises during its CUDA-extension load (mismatched
            # toolchain, missing SM target); treat any import-time failure as
            # "unavailable" rather than crashing the whole registry.
            self._import_error = f"{type(e).__name__}: {e}"
            self._dpa_cls = None

    @property
    def available(self) -> bool:
        """True if ``transformer_engine.pytorch.DotProductAttention`` loaded and CUDA is up."""
        return self._dpa_cls is not None

    def supports(self, shape: AttentionShape, dtype: DType) -> bool:
        return (
            self._dpa_cls is not None
            and dtype in _SUPPORTED_DTYPES
            and shape.head_dim in _SUPPORTED_HEAD_DIMS
            and shape.kind in _SUPPORTED_KINDS
            # Self-attention only for now. TE's DotProductAttention does support
            # cross-attention but needs separate Q vs KV sequence handling that
            # this wrapper does not yet plumb through.
            and shape.is_self_attention
        )

    def _get_module(self, num_heads: int, head_dim: int, causal: bool) -> Any:
        """Return a cached DotProductAttention module for this (H, D, mask) shape."""
        assert self._dpa_cls is not None  # guarded by callers
        mask_type = "causal" if causal else "no_mask"
        key = (num_heads, head_dim, mask_type)
        module = self._cache.get(key)
        if module is None:
            module = self._dpa_cls(
                num_attention_heads=num_heads,
                kv_channels=head_dim,
                attention_dropout=0.0,
                attn_mask_type=mask_type,
            )
            self._cache[key] = module
        return module

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        if self._dpa_cls is None:
            raise RuntimeError(
                "TransformerEngine attention is not available "
                f"(import error: {self._import_error}). "
                "Install `transformer-engine[pytorch]` on a CUDA host, "
                "or pin attention_backend to a ROCm/Triton op. See ADR-0002."
            )
        # Repercep convention is (batch, heads, seq, head_dim) — "bhsd".  TE's
        # DotProductAttention defaults to "sbhd" (seq, batch, heads, dim); we
        # pass qkv_format="bhsd" explicitly so any TE >= 1.7 keeps our layout
        # and no permute is needed.  Older TEs raise on the kwarg and we fall
        # back to permuting in/out.
        _, num_heads, _, head_dim = query.shape
        module = self._get_module(num_heads, head_dim, causal)
        try:
            out: torch.Tensor = module(
                query,
                key,
                value,
                qkv_format="bhsd",
                softmax_scale=scale,
            )
        except TypeError:
            # Older TE: no qkv_format kwarg.  Permute to "sbhd".
            q = query.permute(2, 0, 1, 3).contiguous()
            k = key.permute(2, 0, 1, 3).contiguous()
            v = value.permute(2, 0, 1, 3).contiguous()
            out_sbhd: torch.Tensor = module(q, k, v, softmax_scale=scale)
            out = out_sbhd.permute(1, 2, 0, 3).contiguous()
        return out
