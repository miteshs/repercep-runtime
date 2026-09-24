"""CPU SDPA op — torch.scaled_dot_product_attention dispatched onto oneDNN/AMX.

The "always-available" CPU attention op.  Wraps ``torch.nn.functional
.scaled_dot_product_attention``; on a Sapphire-Rapids-or-newer host with a
recent PyTorch build (>=2.4) the matmuls underneath dispatch to oneDNN's
AMX-tile primitives automatically when the inputs are BF16 or INT8.  This
is the floor for the CPU backend: the BF16-on-AMX path through SDPA is the
"no extension needed" baseline that every later AMX op compares against.

Ordering in :mod:`repercep.attention.registry`:

* :class:`repercep.attention.amx_flash.AMXFlashAttention` — custom C++/AMX
  flash kernel (head_dim ∈ {64,128}, BF16, no mask).  When it qualifies,
  it wins on long S because it does not materialise the QK^T matrix.
* :class:`repercep.attention.ipex_flash.IPEXFlashAttention` — Intel-Extension-
  for-PyTorch's fused attention (broader shape coverage; available only
  when IPEX is installed).
* :class:`AMXSDPAAttention` — the fallback.  Always available; correct on
  any shape; gets AMX wins on BF16 matmul through PyTorch's own oneDNN
  integration.
* :class:`repercep.attention.naive.NaiveAttention` — the FP32 naive floor.

The name is "amx-sdpa" but the op is correct on non-AMX CPUs too; the AMX
prefix advertises *where* the speedup comes from on AMX-class hardware
(oneDNN auto-dispatch), not a hard requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


# torch.nn.functional.scaled_dot_product_attention on CPU supports all
# numeric dtypes Repercep cares about.  We restrict to the ones oneDNN will
# accelerate on AMX (BF16, FP16, INT8) plus FP32 as a correctness floor.
_SUPPORTED_DTYPES = frozenset(
    {DType.FP32, DType.FP16, DType.BF16, DType.INT8}
)

# All Repercep attention kinds are expressible through SDPA's mask + is_causal
# arguments.  Neighborhood attention would need an attn_bias matrix, which
# defeats the AMX matmul fast path; it falls through to a separate op when
# we wire NATTEN-CPU.
_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


class AMXSDPAAttention:
    """torch SDPA on CPU — oneDNN under the hood dispatches to AMX on SPR+."""

    name = "amx-sdpa"

    def __init__(self) -> None:
        self._import_error: str | None = None
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self._import_error = f"torch: {exc}"

    @property
    def available(self) -> bool:
        return self._import_error is None

    def supports(self, shape: AttentionShape, dtype: DType) -> bool:
        return (
            self.available
            and dtype in _SUPPORTED_DTYPES
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
        import torch

        if self._import_error is not None:
            raise RuntimeError(f"AMXSDPAAttention requires torch: {self._import_error}")
        # Repercep convention is (B, H, S, D) — same as SDPA's expected layout
        # on CPU.  No transpose needed (unlike the GPU flash wrappers, which
        # want (B, S, H, D) for their kernels).
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=causal, scale=scale
        )
