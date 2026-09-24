"""FP8 flash-attention via a Triton kernel on MI300X.

This is the AttentionOp wrapper for the fused FP8 flash-attention kernel in
``kernels/triton/fp8_flash_attn.py``.  The kernel uses CDNA3's native FP8 MFMA
instructions (``v_mfma_f32_*_fp8_fp8``) through Triton's ``tl.dot`` with FP8
operands; the running softmax statistics and accumulator stay in FP32.

The fused kernel is the right shape for FP8 attention on long sequences (the
Cosmos DiT runs S up to ~109k tokens, head_dim 128).  The unfused
``fp8_scaled_mm`` path materializes the S^2 scores matrix and is bandwidth-
bound at that scale; the fused tile-by-tile loop streams K/V through
registers and never touches HBM for the scores.

See ADR-0002 and ``kernels/triton/fp8_flash_attn.py`` for the algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


# head_dim values the kernel is compiled for.  The kernel does compile for
# arbitrary D up to the largest tile width on this hardware, but we keep the
# supported set tight to match what's actually tested.
_SUPPORTED_HEAD_DIMS = frozenset({32, 64, 128, 256})

# Triton's tiling needs S divisible by BLOCK_M=128 along the Q axis to skip
# the masked-tail branch; for short S we fall back.  This is a perf cutoff,
# not a correctness one — the kernel masks the tail correctly either way.
_MIN_SEQ_LEN = 128

# Kinds: FULL and CAUSAL.  NEIGHBORHOOD attention is a future kernel.
_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


class FP8TritonAttention:
    """FP8 fused flash-attention via the kernels/ Triton kernel."""

    name = "fp8-triton-flash"

    def __init__(self) -> None:
        self._fn: Any | None = None
        self._import_error: str | None = None
        try:
            # Kernels live outside src/repercep/; importing via package path so
            # `pyproject.toml` doesn't have to ship them.  The kernels/
            # directory must be on sys.path (which it is when invoked from
            # the repo root) — this is the seam between kernel layer and
            # repercep proper.  See kernels/README.md.
            import sys
            from pathlib import Path

            # Resolve to <repo>/kernels.  __file__ is .../src/repercep/attention/fp8_triton.py
            repo_root = Path(__file__).resolve().parents[3]
            kernels_dir = repo_root / "kernels"
            if str(kernels_dir) not in sys.path:
                sys.path.insert(0, str(kernels_dir))

            from triton_kernels.fp8_flash_attn import fp8_flash_attention

            self._fn = fp8_flash_attention
        except ImportError as e:  # pragma: no cover - environment dependent
            self._import_error = str(e)
            self._fn = None
        except Exception as e:  # pragma: no cover - triton compile errors
            # Triton sometimes raises at first call rather than import; we
            # treat any import-time failure as "unavailable" and fall back.
            self._import_error = f"{type(e).__name__}: {e}"
            self._fn = None

    @property
    def available(self) -> bool:
        """True if the Triton FP8 kernel is importable."""
        return self._fn is not None

    def supports(self, shape: AttentionShape, dtype: DType) -> bool:
        return (
            self._fn is not None
            and dtype in (DType.BF16, DType.FP16)
            and shape.head_dim in _SUPPORTED_HEAD_DIMS
            and shape.kind in _SUPPORTED_KINDS
            and shape.seq_len_q >= _MIN_SEQ_LEN
            and shape.seq_len_kv >= _MIN_SEQ_LEN
            # self-attention only for now (Q and KV share scale shape).
            and shape.is_self_attention
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
                "FP8 Triton attention is not available "
                f"(import error: {self._import_error}). "
                "Fall back to attention_backend='naive-sdpa'."
            )
        out: torch.Tensor = self._fn(query, key, value, causal=causal, scale=scale)
        return out
