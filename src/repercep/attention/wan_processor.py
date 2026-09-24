"""Wan-specific attention bridge for diffusers' dispatcher.

Wan's diffusers transformer owns its attention processors, so Repercep's normal
``select_attention_op`` path is not enough to affect model execution.  Recent
diffusers versions route ``WanAttnProcessor`` through
``dispatch_attention_fn(..., backend=self._attention_backend)``; this module
installs that processor with Repercep's registered backend when the existing
``REPERCEP_FP8_ATTENTION`` opt-in is set.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

logger = logging.getLogger(__name__)

_FP8_ENV_VALUES = frozenset(
    {"1", "true", "on", "triton", "scaled_mm", "fa", "flash"}
)

_TRANSFORMER_ATTRS = ("transformer", "transformer_2")


def maybe_install_repercep_wan_attention(pipe: Any) -> bool:
    """Install Repercep attention on a Wan pipeline if the FP8 env var is set.

    Returns ``True`` when at least one Wan transformer accepted the processor.
    Older diffusers releases that do not expose ``WanAttnProcessor`` with the
    dispatch backend seam return ``False`` and keep their native path.
    """
    if os.environ.get("REPERCEP_FP8_ATTENTION", "").lower() not in _FP8_ENV_VALUES:
        return False
    return install_repercep_wan_attention(pipe) > 0


def install_repercep_wan_attention(pipe: Any) -> int:
    """Install Repercep's diffusers backend on all Wan transformers in ``pipe``."""
    processor_cls = _resolve_wan_attn_processor()
    if processor_cls is None:
        return 0

    try:
        from repercep.attention.diffusers_backend import (
            REPERCEP_FP8_BACKEND_NAME,
            register_repercep_fp8_backend,
        )

        backend = register_repercep_fp8_backend()
    except Exception as exc:  # pragma: no cover - diffusers-version dependent
        logger.debug(
            "repercep_wan_attention: backend registration unavailable (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return 0

    try:
        processor = processor_cls()
    except Exception as exc:  # pragma: no cover - diffusers-version dependent
        logger.debug(
            "repercep_wan_attention: WanAttnProcessor unavailable (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return 0

    # diffusers' WanAttnProcessor reads this attribute when calling
    # dispatch_attention_fn.  Keeping the processor class native avoids
    # reimplementing Wan's rotary embedding / normalization details here.
    processor._attention_backend = backend
    processor._repercep_attention_backend = REPERCEP_FP8_BACKEND_NAME

    installed = 0
    for transformer in _iter_wan_transformers(pipe):
        set_attn_processor = getattr(transformer, "set_attn_processor", None)
        if not callable(set_attn_processor):
            logger.debug(
                "repercep_wan_attention: %s has no set_attn_processor",
                type(transformer).__name__,
            )
            continue
        try:
            set_attn_processor(processor)
        except Exception as exc:  # pragma: no cover - model-version dependent
            logger.warning(
                "repercep_wan_attention: failed to install processor on %s (%s: %s)",
                type(transformer).__name__,
                type(exc).__name__,
                exc,
            )
            continue
        installed += 1
    return installed


def _resolve_wan_attn_processor() -> type[Any] | None:
    try:
        from diffusers.models.transformers.transformer_wan import WanAttnProcessor
    except Exception as exc:  # pragma: no cover - diffusers-version dependent
        logger.debug(
            "repercep_wan_attention: WanAttnProcessor unavailable (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return None
    return cast("type[Any]", WanAttnProcessor)


def _iter_wan_transformers(pipe: Any) -> tuple[Any, ...]:
    transformers: list[Any] = []
    for attr in _TRANSFORMER_ATTRS:
        transformer = getattr(pipe, attr, None)
        if transformer is not None:
            transformers.append(transformer)
    return tuple(transformers)


__all__ = [
    "install_repercep_wan_attention",
    "maybe_install_repercep_wan_attention",
]
