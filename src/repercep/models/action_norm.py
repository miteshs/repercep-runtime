"""Quantile action denormalization — the formula three ports now share.

Every action-chunk model we have ported normalizes its action channels to
``[-1, 1]`` against per-channel ``q01``/``q99`` quantiles and expects the
caller to invert it:

    ``x_real = (x_norm + 1) / 2 * (q99 - q01) + q01``

LingBot-VA (``lingbot_va_pipeline._denormalize_actions``), DreamZero
(``dreamzero_pipeline._denormalize_actions``) and now Cosmos 3 all implement
it. The two existing implementations differ in one detail that is *not*
cosmetic — LingBot-VA adds ``1e-6`` to the denominator span and DreamZero does
not — so this helper takes ``eps`` explicitly rather than picking a winner.

**Deliberately not refactoring the two existing callers.** Both were verified
against their reference stacks on GPU (exact-parity gates, see
``docs/LINGBOT_VA_SEAM_VERIFY.md`` and ``docs/DREAMZERO_PORT_PLAN.md`` §2b),
and swapping a numerically-sensitive helper underneath a verified path without
a GPU to re-verify on would trade a real guarantee for tidiness. Migrate them
when a pod is up and the parity gates can be re-run — see
``docs/COSMOS3_PORT_PLAN.md`` §3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def denormalize_quantile(
    actions: torch.Tensor,
    q01: torch.Tensor,
    q99: torch.Tensor,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    """Invert symmetric per-channel quantile normalization.

    ``actions`` is ``(..., C)`` in ``[-1, 1]``; ``q01``/``q99`` are ``(C,)``
    per-channel quantiles broadcast over the leading dimensions. ``eps`` pads
    the span denominator — pass the value the reference implementation being
    matched uses (``1e-6`` for the LingBot-VA lineage, ``0.0`` for DreamZero
    and Cosmos 3), because a mismatch here shows up as a small, plausible-
    looking, entirely wrong trajectory rather than as an error.
    """
    if q01.shape != q99.shape:
        raise ValueError(f"q01 shape {tuple(q01.shape)} != q99 shape {tuple(q99.shape)}")
    if actions.shape[-1] != q01.shape[-1]:
        raise ValueError(
            f"actions have {actions.shape[-1]} channels but stats have {q01.shape[-1]}"
        )
    q01 = q01.to(device=actions.device, dtype=actions.dtype)
    q99 = q99.to(device=actions.device, dtype=actions.dtype)
    return (actions + 1) / 2 * (q99 - q01 + eps) + q01
