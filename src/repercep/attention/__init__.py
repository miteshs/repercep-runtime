"""Attention primitives for Repercep.

World models lean hard on attention: DiT blocks use full bidirectional spatial
attention, temporal layers use causal attention, and video models often use
NATTEN-style neighborhood attention.  Each op implements the ``AttentionOp``
Protocol; ``select_attention_op`` picks the fastest one that supports a given
problem shape on a given architecture.

On MI300X the Hopper-only FlashAttention-3 kernels do not apply (no wgmma/TMA);
the equivalent is the Composable-Kernel ``flash-attn`` build or aotriton-backed
SDPA.  See ADR-0002.
"""

from __future__ import annotations

# Importing the diffusers-side bridge here registers the ``"repercep_fp8"``
# backend with ``diffusers.models.attention_dispatch._AttentionBackendRegistry``
# at package import time. The dispatcher stays on ``"native"`` unless the user
# flips it (env var or ``attention_backend(...)`` context manager) so this is
# zero-cost in the default path. See ``diffusers_backend.py``.
from repercep.attention import diffusers_backend as _diffusers_backend  # noqa: F401
from repercep.attention.fp8_hopper_triton import FP8HopperTritonAttention
from repercep.attention.fp8_scaled_mm import FP8ScaledMMAttention
from repercep.attention.fp8_triton import FP8TritonAttention
from repercep.attention.protocol import AttentionOp
from repercep.attention.registry import select_attention_op
from repercep.attention.types import AttentionKind, AttentionShape

__all__ = [
    "AttentionKind",
    "AttentionOp",
    "AttentionShape",
    "FP8HopperTritonAttention",
    "FP8ScaledMMAttention",
    "FP8TritonAttention",
    "select_attention_op",
]
