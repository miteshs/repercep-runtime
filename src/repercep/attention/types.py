"""Shape and kind descriptors for attention problems."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class AttentionKind(enum.StrEnum):
    """The attention pattern a layer needs.

    World models exercise all three: ``FULL`` for DiT spatial blocks,
    ``CAUSAL`` for temporal autoregression, ``NEIGHBORHOOD`` for NATTEN-style
    local windowed attention over video latents.
    """

    FULL = "full"
    CAUSAL = "causal"
    NEIGHBORHOOD = "neighborhood"


@dataclass(frozen=True, slots=True)
class AttentionShape:
    """The shape of a single attention call.

    Tensor layout convention across Repercep is ``(batch, heads, seq, head_dim)``
    (SDPA-style); ops that need a different layout (e.g. ``flash-attn`` wants
    ``(batch, seq, heads, head_dim)``) transpose internally and document it.
    """

    batch: int
    heads: int
    seq_len_q: int
    seq_len_kv: int
    head_dim: int
    kind: AttentionKind = AttentionKind.FULL

    @property
    def is_self_attention(self) -> bool:
        return self.seq_len_q == self.seq_len_kv
