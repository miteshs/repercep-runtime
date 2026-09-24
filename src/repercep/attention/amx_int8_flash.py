"""Custom AMX INT8 flash-attention — the Repercep-owned CPU kernel.

Sibling of :class:`repercep.attention.amx_flash.AMXFlashAttention` (BF16) and
:class:`repercep.attention.fp8_triton.FP8TritonAttention` (gfx942).  Same
per-vendor, per-silicon pattern as the rest of the registry:
one source file per ISA, parameterised over shape but never over arch.  On
Intel SPR+ the kernel uses AMX_INT8 (TDPBSSD) for the QK^T and PV matmuls
inside a flash-attention online-softmax loop, with AVX-512 BF16 driving the
per-tile dynamic activation quant and the FP32 softmax + scale steps.

The kernel itself lives in ``kernels/cpu/amx_int8_attn/`` as a torch C++
extension.  Build is via setuptools' BuildExtension on demand, so a host
without ``amx_int8`` doesn't pay any compile cost (the :meth:`available`
probe returns False before it tries to import).

The wrapper mirrors :class:`AMXFlashAttention` exactly — same
``available`` / ``supports`` / ``__call__`` contract — so the registry's
INTEL branch needs no special-casing.

Surface contract:
    * Caller hands BF16 Q/K/V of shape ``(B, H, S, D)``.
    * Kernel dynamically quantizes Q/K/V to INT8 internally (per-token
      symmetric for Q/K/P, per-tile symmetric for V).
    * Output is BF16, same shape.
    * ``head_dim`` must be in ``{64, 128}``; ``S`` must be a positive
      multiple of 32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


# Head dims the AMX INT8 kernel ships compiled tiles for.  AMX INT8 tile
# rows are 16-wide and columns are 64-byte (= 64 int8 lanes).  Head dims
# that are multiples of 64 hit the d_chunk path cleanly; the two values
# Cosmos / Wan use today (64 and 128) are the supported set.
_SUPPORTED_HEAD_DIMS = frozenset({64, 128})

# AMX_INT8 is the lever; the caller still hands us BF16 (input contract)
# and we quantize internally.  Only BF16 is advertised because that's the
# input dtype the kernel checks for in C++.  INT8-on-the-input would
# require a different quant-scheme contract and is a future surface.
_SUPPORTED_DTYPES = frozenset({DType.BF16})

_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


def _detect_amx_int8() -> bool:
    """True iff /proc/cpuinfo reports the amx_int8 feature flag."""
    try:
        for line in open("/proc/cpuinfo").read().splitlines():  # noqa: SIM115
            if not line.startswith("flags"):
                continue
            return "amx_int8" in line.split()
    except OSError:
        return False
    return False


class AMXInt8FlashAttention:
    """Flash-attention on Intel AMX_INT8.  Repercep-owned C++ kernel.

    BF16-in / BF16-out: the caller does not see INT8.  Quantization is
    dynamic and per-tile, mirroring the per-channel symmetric scheme used
    by :mod:`repercep.runtime.quantize` for static weight quant.
    """

    name = "amx-int8-flash"

    def __init__(self) -> None:
        self._fn: Any | None = None
        self._import_error: str | None = None
        if not _detect_amx_int8():
            self._import_error = "/proc/cpuinfo missing amx_int8 flag"
            return
        try:
            # The extension module lives next to the BF16 sibling.  Built
            # via ``python setup.py build_ext --inplace`` in
            # ``kernels/cpu/amx_int8_attn/``.
            from kernels.cpu.amx_int8_attn._native import flash_attn_int8

            self._fn = flash_attn_int8
        except ImportError as exc:
            self._import_error = f"kernels.cpu.amx_int8_attn._native: {exc}"

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
                f"AMXInt8FlashAttention not available: {self._import_error}.  "
                "Build the kernel with `make kernels-cpu-int8` or fall back "
                "to the AMX BF16 sibling / AMXSDPAAttention floor."
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
