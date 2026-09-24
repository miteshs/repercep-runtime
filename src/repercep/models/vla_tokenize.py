"""Action (de)tokenization for token-VLA engines — CPU-pure, unit-tested.

A token VLA discretizes each continuous action dimension into ``bins`` buckets
over a per-dimension ``[low, high]`` range (OpenVLA: 256 bins over the training
dataset's ``q01``/``q99`` stats) and decodes an action token back to its
bucket's value. This module is that math, isolated so it is verified on CPU
without weights — the same "catch the silent reshape/quantization bug cheaply,
not on a GPU pod" discipline as ``lingbot_va_pipeline``'s flatten/unflatten
helpers.

Honesty note (``docs/VLA_PORT_PLAN.md`` §5): the *exact* index convention
(``np.digitize`` against bin edges vs. bin-center rounding, and the -1 offset
OpenVLA applies because token id 0 is reserved) is reconciled against the
model's own code in Phase 1. This is the general **uniform-binning** form; the
roundtrip property tested here — decode∘encode is identity to within one bin —
holds regardless of that convention, so it is the right thing to pin down now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _safe_span(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """``high - low``, with zero-width dims (constant action channel) mapped to
    1.0 so the division is well-defined; such a dim always decodes to ``low``."""
    import torch

    span = high - low
    return torch.where(span == 0, torch.ones_like(span), span)


def discretize_actions(
    actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor, bins: int
) -> torch.Tensor:
    """Continuous actions ``(..., D)`` → integer action-token indices ``(..., D)``.

    Values are clamped to ``[low, high]`` (out-of-distribution actions saturate
    at the end buckets rather than wrapping), then mapped to ``0..bins-1``.
    ``low``/``high`` are per-dimension and broadcast over the leading dims.
    """
    import torch

    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    clipped = torch.minimum(torch.maximum(actions, low), high)
    frac = (clipped - low) / _safe_span(low, high)  # [0, 1]
    idx = torch.round(frac * (bins - 1)).to(torch.long)
    return idx.clamp_(0, bins - 1)


def undiscretize_actions(
    tokens: torch.Tensor, low: torch.Tensor, high: torch.Tensor, bins: int
) -> torch.Tensor:
    """Action-token indices ``(..., D)`` → continuous actions ``(..., D)``.

    Each index decodes to its bucket value on the uniform grid over
    ``[low, high]`` — the inverse of :func:`discretize_actions` to within one
    bin's resolution.
    """

    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    frac = tokens.to(low.dtype) / (bins - 1)
    return low + frac * (high - low)


__all__ = ["discretize_actions", "undiscretize_actions"]
