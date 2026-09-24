"""Custom AMX flash-attention — the Repercep-owned CPU kernel.

Sibling of :class:`repercep.attention.fp8_triton.FP8TritonAttention` (gfx942)
and :class:`repercep.attention.fp8_hopper_triton.FP8HopperTritonAttention`
(sm_90a): per-vendor kernels, one source per silicon, parameterised over
shape but never over arch.  On Intel the kernel uses AMX_BF16 (TDPBF16PS)
for the QK^T and PV matmuls inside a flash-attention online-softmax loop,
and AVX-512 (BF16 fast-math) for the softmax + scale steps.

The kernel itself lives in ``kernels/cpu/amx_attn/`` as a torch C++
extension (CPU-side counterpart of the Triton kernels in
``kernels/triton_kernels/``).  Build is via setuptools' BuildExtension on
demand, so a host without ``amx_bf16`` doesn't pay any compile cost (the
:meth:`available` probe returns False before it tries to import).

The wrapper here mirrors the GPU flash wrappers' surface area exactly — same
``available`` / ``supports`` / ``__call__`` contract — so the registry's
INTEL branch needs no special-casing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


# Head dims the AMX kernel ships compiled tiles for.  AMX tile rows are
# 16-wide and columns 64-byte; for BF16 (2 bytes) that's 32 elements per
# tile column.  Head dims that are multiples of 32 hit the fast path
# cleanly; 64 and 128 are the only ones Cosmos / Wan actually use today,
# so those are the ones we compile.
_SUPPORTED_HEAD_DIMS = frozenset({64, 128})

# AMX_BF16 is the lever; FP16 falls through to the AVX-512_FP16 path which
# is slower than BF16 on SPR (no AMX_FP16 until Granite Rapids).  We only
# advertise BF16 to keep the dispatch contract honest.
_SUPPORTED_DTYPES = frozenset({DType.BF16})

_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


def _detect_amx_bf16() -> bool:
    """True iff /proc/cpuinfo reports the amx_bf16 feature flag."""
    try:
        for line in open("/proc/cpuinfo").read().splitlines():  # noqa: SIM115
            if not line.startswith("flags"):
                continue
            return "amx_bf16" in line.split()
    except OSError:
        return False
    return False


class AMXFlashAttention:
    """Flash-attention on Intel AMX_BF16.  Repercep-owned C++ kernel."""

    name = "amx-bf16-flash"

    def __init__(self) -> None:
        self._fn: Any | None = None
        self._import_error: str | None = None
        if not _detect_amx_bf16():
            self._import_error = "/proc/cpuinfo missing amx_bf16 flag"
            return
        try:
            # The extension module lives next to the Triton kernels.  Built
            # via ``python setup.py build_ext --inplace`` in
            # ``kernels/cpu/amx_attn/`` (mirrors the HIP scaffold in
            # ``kernels/hip/``).
            from kernels.cpu.amx_attn._native import flash_attn_bf16

            self._fn = flash_attn_bf16
        except ImportError as exc:
            self._import_error = f"kernels.cpu.amx_attn._native: {exc}"

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
        import math

        if self._fn is None:
            raise RuntimeError(
                f"AMXFlashAttention not available: {self._import_error}.  "
                "Build the kernel with `make kernels-cpu` or fall back to "
                "the AMXSDPAAttention floor."
            )
        # Repercep convention: (B, H, S, D).  The C++ kernel takes the same
        # layout, contiguous in S.  No transpose; the contiguous() call is
        # cheap (no-op when already contiguous) and the kernel's loads
        # assume row-major over S.
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        sm_scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        return self._fn(q, k, v, sm_scale, causal)  # type: ignore[no-any-return]
