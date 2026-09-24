"""Custom AMX FP16 flash-attention — the Repercep-owned Granite Rapids kernel.

Granite Rapids (GNR) sibling of :class:`repercep.attention.amx_flash.AMXFlashAttention`
(which is the AMX_BF16 path on Sapphire/Emerald Rapids).  GNR is the first
Intel silicon that adds ``AMX_FP16`` (``TDPFP16PS``) to the SPR/EMR baseline,
so this is the FP16 fast-path for that generation.  SPR and EMR carry only
AMX_BF16 and so disqualify themselves here -- they continue to run FP16 via
the AVX-512_FP16 SDPA floor (no AMX involvement).

The kernel itself lives in ``kernels/cpu/amx_fp16_attn/`` as a torch C++
extension.  As of Item C of the CPU port the kernel is **scaffolded** rather
than finished; the wrapper still exposes the full surface area so the
attention registry and backend layer can wire it up uniformly.  On the
current dev hosts (SPR with masked AMX) :meth:`available` is ``False`` --
the same end-state as the BF16 sibling on a non-AMX host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


# Head dims the AMX kernel ships compiled tiles for.  AMX tile rows are
# 16-wide and columns 64-byte; for FP16 (2 bytes) that's 32 elements per
# tile column -- byte-identical to BF16.  64 and 128 are the only head_dims
# Cosmos / Wan exercise today.
_SUPPORTED_HEAD_DIMS = frozenset({64, 128})

# AMX_FP16 is the GNR-only lever this wrapper targets.  BF16 inputs belong
# to the sibling :class:`AMXFlashAttention` -- keeping the dispatch contract
# clean means each wrapper advertises exactly one dtype.
_SUPPORTED_DTYPES = frozenset({DType.FP16})

_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


def _detect_amx_fp16() -> bool:
    """True iff /proc/cpuinfo reports the amx_fp16 feature flag.

    Mirrors :func:`repercep.attention.amx_flash._detect_amx_bf16` but probes
    the GNR-specific flag.  SPR/EMR report ``amx_bf16`` but not
    ``amx_fp16``, so this returns ``False`` on both -- which is the correct
    routing for the FP16 path on those generations (they fall through to
    the AVX-512_FP16 SDPA floor).
    """
    try:
        for line in open("/proc/cpuinfo").read().splitlines():  # noqa: SIM115
            if not line.startswith("flags"):
                continue
            return "amx_fp16" in line.split()
    except OSError:
        return False
    return False


class AMXFP16FlashAttention:
    """Flash-attention on Intel AMX_FP16 (Granite Rapids).  Repercep-owned C++ kernel."""

    name = "amx-fp16-flash"

    def __init__(self) -> None:
        self._fn: Any | None = None
        self._import_error: str | None = None
        if not _detect_amx_fp16():
            self._import_error = "/proc/cpuinfo missing amx_fp16 flag"
            return
        try:
            # The extension module lives next to the BF16 sibling.  Built
            # via ``python setup.py build_ext --inplace`` in
            # ``kernels/cpu/amx_fp16_attn/``.  setup.py refuses to build
            # without AMX_FP16, so reaching this import on a non-GNR host
            # is already impossible.
            from kernels.cpu.amx_fp16_attn._native import flash_attn_fp16

            self._fn = flash_attn_fp16
        except ImportError as exc:
            self._import_error = f"kernels.cpu.amx_fp16_attn._native: {exc}"

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
                f"AMXFP16FlashAttention not available: {self._import_error}.  "
                "Build the kernel with `make kernels-cpu` on a Granite Rapids "
                "host, or fall back to the AMXSDPAAttention floor."
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
