"""Reference attention via torch SDPA.

Correct on every backend and dtype.  This is the measurement floor: the
benchmark harness reports speedups relative to a model running entirely on
``NaiveAttention``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from repercep.attention.types import AttentionShape
    from repercep.hardware import DType


class NaiveAttention:
    """``torch.nn.functional.scaled_dot_product_attention`` wrapped as an ``AttentionOp``.

    On ROCm, SDPA itself dispatches to aotriton-compiled flash kernels when the
    shape qualifies, so this is not as slow as the name implies — but Repercep
    treats it as the correctness reference, not the optimized path.
    """

    name = "naive-sdpa"
    available = True

    def supports(self, shape: AttentionShape, dtype: DType) -> bool:
        return True

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        from torch.nn.functional import scaled_dot_product_attention

        return scaled_dot_product_attention(query, key, value, is_causal=causal, scale=scale)
