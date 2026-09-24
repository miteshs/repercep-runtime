"""FP8 flash-attention via a Triton kernel on NVIDIA Ada Lovelace (sm_89).

This is the AttentionOp wrapper for the fused FP8 flash-attention kernel in
``kernels/triton_kernels/fp8_flash_attn_ada.py``.  The kernel uses Ada's
native FP8 mma.sync tensor-core instructions
(``mma.sync.aligned.m16n8k32.f32.e4m3.e4m3``) through Triton's ``tl.dot``
with FP8 operands; the running softmax statistics and accumulator stay in
FP32.

This is the Ada Lovelace sibling of ``fp8_hopper_triton.py`` (sm_90a) and
``fp8_triton.py`` (gfx942 / MI300X).  Algorithm is identical (FlashAttention-2
online softmax); the differences are:

- Underlying tensor-core instruction: ``mma.sync`` (synchronous) on Ada,
  ``wgmma.mma_async`` on Hopper, MFMA on CDNA3.
- FP8 dtype: ``e4m3fn`` (IEEE) on both Ada and Hopper, ``e4m3fnuz`` on
  CDNA3.  FP8 range: 448 (Ada/Hopper), 240 (CDNA3).
- Autotune grid: Ada has 100 KiB SMEM/block (vs Hopper's 228 KiB and
  CDNA3's 64 KiB), so we cap tiles at 256x128 and skip the 256x256 entry
  that the Hopper kernel uses.  Ada's narrower SMs prefer num_warps=4
  more often than Hopper.
- num_stages: Ada's synchronous mma.sync doesn't benefit from deep
  pipelining the way Hopper's wgmma does; sweep (2, 3) rather than
  (2, 3, 4, 5).

Strategic context.  Ada FP8 ISA is real (introduced with the L40 / RTX
4090 / RTX 6000 Ada generation in late 2022) but rarely used in practice
because PyTorch's FP8 ecosystem (transformer_engine, FA-3 FP8) was built
Hopper-first.  This wrapper gives Repercep a fourth silicon target for the
FP8 attention lever without requiring Transformer Engine or FA-3.

This wrapper disqualifies on any device cap other than (8, 9) — sm_86
(Ampere consumer) and sm_80 (A100) do NOT have FP8 tensor cores; sm_90
(Hopper) has its own dedicated kernel.

See ADR-0002 and ``kernels/triton_kernels/fp8_flash_attn_ada.py`` for
the algorithm.
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

# Compute capability gate.  Ada Lovelace is the only NVIDIA arch where this
# kernel makes sense:
# - sm_80 (A100) and sm_86 (RTX 3090 / RTX A6000): no FP8 tensor cores; the
#   kernel would compile but emit emulated FP8 mma that's slower than BF16.
# - sm_89 (Ada Lovelace, RTX 4090 / L40 / L40S / RTX 2000-6000 Ada): native
#   FP8 mma.sync — this kernel's target.
# - sm_90 (Hopper): use fp8_hopper_triton instead; it emits warpgroup
#   wgmma which is 2-3x faster than mma.sync on the same FP8 inputs.
_REQUIRED_DEVICE_CAPABILITY = (8, 9)


def _device_is_ada() -> bool:
    """Return True iff the current CUDA device is Ada Lovelace (sm_89).

    Returns False on CPU-only hosts, non-NVIDIA GPUs (CUDA not available),
    and on any other compute capability.  This is the silicon gate — if
    False, the op disqualifies in ``supports`` and ``available``.
    """
    try:
        import torch as _torch  # local import to keep module-level fast.

        if not _torch.cuda.is_available():
            return False
        return _torch.cuda.get_device_capability(0) == _REQUIRED_DEVICE_CAPABILITY
    except Exception:
        # Defensive: any import / probe failure → unavailable.  We never
        # want this gate to crash the registry on a host that doesn't
        # have torch.cuda set up.
        return False


class FP8AdaTritonAttention:
    """FP8 fused flash-attention via the kernels/ Triton kernel on Ada Lovelace."""

    name = "fp8-ada-triton-flash"

    def __init__(self) -> None:
        self._fn: Any | None = None
        self._import_error: str | None = None

        # Silicon gate first — if we're not on Ada, don't even import the
        # kernel.  This keeps the registry probe cheap and avoids
        # Triton-compile errors on the wrong arch.
        if not _device_is_ada():
            self._import_error = (
                "FP8 Ada Triton attention disqualified: device capability is not (8, 9). "
                "Use fp8-hopper-triton-flash (sm_90), fp8-triton-flash (gfx942), or naive-sdpa."
            )
            return

        try:
            # Kernels live outside src/repercep/; importing via package path so
            # `pyproject.toml` doesn't have to ship them.  The kernels/
            # directory must be on sys.path (which it is when invoked from
            # the repo root) — this is the seam between kernel layer and
            # repercep proper.  See kernels/README.md.
            import sys
            from pathlib import Path

            # Resolve to <repo>/kernels.  __file__ is
            # .../src/repercep/attention/fp8_ada_triton.py
            repo_root = Path(__file__).resolve().parents[3]
            kernels_dir = repo_root / "kernels"
            if str(kernels_dir) not in sys.path:
                sys.path.insert(0, str(kernels_dir))

            from triton_kernels.fp8_flash_attn_ada import fp8_flash_attention

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
        """True if we're on Ada Lovelace AND the Triton FP8 kernel is importable."""
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
                "FP8 Ada Triton attention is not available "
                f"(import error: {self._import_error}). "
                "Fall back to attention_backend='naive-sdpa'."
            )
        out: torch.Tensor = self._fn(query, key, value, causal=causal, scale=scale)
        return out


__all__ = ["FP8AdaTritonAttention"]
