"""Bridge Repercep's AMX attention into diffusers' attention dispatcher.

Sibling of :mod:`repercep.attention.diffusers_backend` (which registers
``"repercep_fp8"`` for the GPU path).  This module registers
``"repercep_amx"`` with diffusers' ``_AttentionBackendRegistry`` so the
Cosmos diffusers pipeline can route through the AMX-aware op on CPU.

Routing logic mirrors the FP8 path exactly:

* head_dim in the kernel's compiled set ({64, 128})
* self-attention (Q.shape == K.shape == V.shape)
* BF16 dtype
* sequence long enough that the flash-attention memory savings beat
  the fused-kernel dispatch overhead

Below the crossover the call falls back to torch SDPA on CPU (which
dispatches to oneDNN/AMX automatically for BF16 matmul) — the safe
default.

Activation: ``maybe_activate_from_env`` reads ``REPERCEP_AMX_ATTENTION``.
With it set, the diffusers dispatcher routes Cosmos attention through
``_repercep_amx_attention``; unset, the dispatcher stays on ``"native"``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


REPERCEP_AMX_BACKEND_NAME = "repercep_amx"


# Same shape gating logic as the FP8 GPU bridge: the flash kernel's
# memory-pattern win only kicks in above the seq_len where the dispatch
# overhead is amortised.  Conservatively set to the same 4 k tokens
# crossover the FP8 path uses (the kernel's per-call dispatch overhead
# is dominated by the same factors).
_AMX_MIN_SEQ_LEN = 4096

# head_dim values the AMX BF16 kernel is compiled for.  Mirrors
# ``AMXFlashAttention._SUPPORTED_HEAD_DIMS``.
_AMX_SUPPORTED_HEAD_DIMS = frozenset({64, 128})


def _route_should_use_amx(query: Any, key: Any, value: Any) -> bool:
    """Decide whether to dispatch this call to the AMX flash kernel.

    Diffusers passes ``(batch, seq_len, num_heads, head_dim)``.  The AMX
    kernel takes the same layout the FP8 GPU sibling does, and applies
    the same constraints: self-attention only, head_dim in the compiled
    set, sequence above the crossover.  BF16 only (the AMX op disqualifies
    FP16 internally because there is no AMX_FP16 on Sapphire Rapids).
    """
    import torch

    if query.dtype is not torch.bfloat16:
        return False
    if query.shape != key.shape or query.shape != value.shape:
        return False
    head_dim = query.shape[-1]
    if head_dim not in _AMX_SUPPORTED_HEAD_DIMS:
        return False
    seq_len = query.shape[-3]
    return bool(seq_len >= _AMX_MIN_SEQ_LEN)


def _native_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """Same shape-shuffling fallback as the FP8 GPU bridge."""
    import torch

    if (
        attn_mask is not None
        and attn_mask.ndim == 2
        and attn_mask.shape[0] == query.shape[0]
        and attn_mask.shape[1] == key.shape[1]
    ):
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

    q_bhsd = query.permute(0, 2, 1, 3)
    k_bhsd = key.permute(0, 2, 1, 3)
    v_bhsd = value.permute(0, 2, 1, 3)
    out = torch.nn.functional.scaled_dot_product_attention(
        query=q_bhsd,
        key=k_bhsd,
        value=v_bhsd,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return out.permute(0, 2, 1, 3)


def _repercep_amx_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Any = None,
) -> torch.Tensor:
    """The diffusers-registered ``"repercep_amx"`` backend function."""
    import torch

    if return_lse:
        raise ValueError("repercep_amx backend does not support return_lse=True.")
    if _parallel_config is not None:
        return _native_fallback(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
        )

    if (
        attn_mask is not None
        or dropout_p != 0.0
        or enable_gqa
        or not _route_should_use_amx(query, key, value)
    ):
        return _native_fallback(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
        )

    # Lazy import — the op holds a C++ extension load and we don't want
    # to pay it when the backend isn't active.
    op = getattr(_repercep_amx_attention, "_op", None)
    if op is None:
        from repercep.attention.amx_flash import AMXFlashAttention

        op = AMXFlashAttention()
        _repercep_amx_attention._op = op  # type: ignore[attr-defined]
    if not op.available:
        return _native_fallback(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
        )

    # diffusers (B, S, H, D) -> kernel (B, H, S, D), then back.
    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    out_bhsd = op(q_bhsd, k_bhsd, v_bhsd, causal=is_causal, scale=scale)
    if not torch.isfinite(out_bhsd).all():
        logger.warning(
            "repercep_amx: non-finite output at shape %s; falling back to native SDPA",
            tuple(query.shape),
        )
        return _native_fallback(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
        )
    return out_bhsd.permute(0, 2, 1, 3).type_as(query)


def _ensure_backend_name_registered() -> Any:
    """Same enum-member injection as the FP8 bridge."""
    from diffusers.models.attention_dispatch import AttentionBackendName

    if REPERCEP_AMX_BACKEND_NAME in AttentionBackendName._value2member_map_:
        return AttentionBackendName._value2member_map_[REPERCEP_AMX_BACKEND_NAME]
    new_member = str.__new__(AttentionBackendName, REPERCEP_AMX_BACKEND_NAME)
    new_member._name_ = "REPERCEP_AMX"
    new_member._value_ = REPERCEP_AMX_BACKEND_NAME
    AttentionBackendName._member_map_["REPERCEP_AMX"] = new_member
    AttentionBackendName._value2member_map_[REPERCEP_AMX_BACKEND_NAME] = new_member
    if "REPERCEP_AMX" not in AttentionBackendName._member_names_:
        AttentionBackendName._member_names_.append("REPERCEP_AMX")
    return new_member


def register_repercep_amx_backend() -> Any:
    """Register the ``"repercep_amx"`` backend with diffusers' dispatcher."""
    import inspect

    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    member = _ensure_backend_name_registered()
    if member in _AttentionBackendRegistry._backends:
        return member

    _AttentionBackendRegistry._backends[member] = _repercep_amx_attention
    _AttentionBackendRegistry._constraints[member] = []
    _AttentionBackendRegistry._supported_arg_names[member] = set(
        inspect.signature(_repercep_amx_attention).parameters.keys()
    )
    return member


def activate_repercep_amx_backend() -> bool:
    """Make ``"repercep_amx"`` the active backend for diffusers' dispatcher."""
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    member = register_repercep_amx_backend()
    if _AttentionBackendRegistry._active_backend == member:
        return False
    _AttentionBackendRegistry.set_active_backend(member)
    return True


def maybe_activate_from_env() -> bool:
    """Activate iff ``REPERCEP_AMX_ATTENTION`` is set to a truthy value."""
    amx_env = os.environ.get("REPERCEP_AMX_ATTENTION", "").lower()
    if amx_env not in ("1", "true", "on", "amx", "ipex"):
        return False
    if amx_env == "ipex":
        # IPEX has its own diffusers integration; the AMX bridge would
        # double-dispatch.  Log and skip so the user's env var isn't
        # silently ignored.
        logger.info(
            "repercep_amx: REPERCEP_AMX_ATTENTION=ipex — the diffusers bridge "
            "only routes through the Repercep AMX kernel; use IPEX's own "
            "diffusers integration for that path."
        )
        return False
    return activate_repercep_amx_backend()


try:
    register_repercep_amx_backend()
except Exception as exc:  # pragma: no cover - environment dependent
    logger.debug("repercep_amx: registration deferred (%s: %s)", type(exc).__name__, exc)


__all__ = [
    "REPERCEP_AMX_BACKEND_NAME",
    "activate_repercep_amx_backend",
    "maybe_activate_from_env",
    "register_repercep_amx_backend",
]
