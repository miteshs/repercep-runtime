"""Attention op selection."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from repercep.attention.amx_sdpa import AMXSDPAAttention
from repercep.attention.fp8_hopper_triton import FP8HopperTritonAttention
from repercep.attention.fp8_scaled_mm import FP8ScaledMMAttention
from repercep.attention.fp8_triton import FP8TritonAttention
from repercep.attention.hopper_flash import HopperFlashAttention
from repercep.attention.naive import NaiveAttention
from repercep.attention.rocm_flash import ROCmFlashAttention
from repercep.attention.transformer_engine import TransformerEngineAttention
from repercep.hardware import DeviceArch, DType, Vendor

if TYPE_CHECKING:
    from repercep.attention.protocol import AttentionOp
    from repercep.attention.types import AttentionShape


# FP8 is opt-in via env var until the perf bench across all Cosmos shapes
# bears out the universal win.  At long S (~32k+) the Triton FP8 flash
# kernel beats SDPA->aotriton; at short S the dispatch + quantization overhead
# wins.  See docs/OPTIMIZATION.md and docs/BUILD_LOG.md for the measured
# crossover.
#
# Env values:
#   1 / true / on    — auto-pick the best FP8 op for the host's vendor
#   triton           — force the Repercep Triton FP8 kernel (AMD: gfx942 tile;
#                      NVIDIA: Hopper-tuned sibling kernel)
#   scaled_mm        — force the AMD-only ``torch._scaled_grouped_mm`` path
#   te / transformer_engine
#                    — force the NVIDIA TransformerEngine FP8 path
#                      (Hopper, optional install)
_FP8_ENV = os.environ.get("REPERCEP_FP8_ATTENTION", "").lower()
_FP8_TRUTHY = ("1", "true", "on", "triton", "scaled_mm", "te", "transformer_engine")
_FP8_ENABLED = _FP8_ENV in _FP8_TRUTHY

# AMX is opt-in via the parallel env var so the CPU default stays at the
# SDPA->oneDNN floor (always-available, correct on every shape).  Setting
# REPERCEP_AMX_ATTENTION=1 promotes the AMX-aware flash op when the shape
# qualifies (head_dim ∈ {64, 128}, BF16, no mask) and IPEX when present;
# otherwise SDPA on CPU still gets AMX wins via oneDNN's auto-dispatch.
#
# Env values:
#   1 / true / on        — auto-pick best AMX op for the host (BF16 wins for
#                          BF16 calls; FP16 wins for FP16 on Granite Rapids)
#   amx                  — force the Repercep AMX BF16 kernel (SPR/EMR)
#   int8                 — force the Repercep AMX INT8 kernel (SPR/EMR, dynamic
#                          per-tile activation quant; BF16 in / BF16 out)
#   fp16                 — force the Repercep AMX FP16 kernel (Granite Rapids+)
#   ipex                 — force the IPEX fused attention
#
# Note on auto-pick: INT8 is NOT in the auto-set because its per-tile dynamic
# quantization has its own shape crossover (not yet measured on real AMX
# silicon — Session 17 close documents the build is hardware-blocked).  It
# stays opt-in until that crossover lands.
_AMX_ENV = os.environ.get("REPERCEP_AMX_ATTENTION", "").lower()
_AMX_TRUTHY = ("1", "true", "on", "amx", "int8", "fp16", "ipex")
_AMX_ENABLED = _AMX_ENV in _AMX_TRUTHY


def select_attention_op(arch: DeviceArch, shape: AttentionShape, dtype: DType) -> AttentionOp:
    """Pick the fastest attention op that supports ``(shape, dtype)`` on ``arch``.

    Order: vendor-specific fused ops (when env-enabled and shape qualifies)
    -> vendor flash / SDPA -> naive.  Selection is by problem shape, so a
    model can use the FP8 kernel for its big DiT blocks, the flash kernel
    for medium ones, and the naive floor for an unusual head_dim — no
    caller-side branching.
    """
    candidates: list[AttentionOp] = []
    if arch.vendor is Vendor.AMD:
        if _FP8_ENABLED and _FP8_ENV not in ("scaled_mm", "te", "transformer_engine"):
            # The fused Triton kernel — preferred when it qualifies.  Its
            # supports() already gates on seq_len >= 128 etc.
            candidates.append(FP8TritonAttention())
        if _FP8_ENABLED and _FP8_ENV not in ("triton", "te", "transformer_engine"):
            candidates.append(FP8ScaledMMAttention())
        candidates.append(ROCmFlashAttention())
    elif arch.vendor is Vendor.NVIDIA:
        # NVIDIA FP8 order: Triton kernel (the Repercep-owned path, sibling of
        # the AMD one) -> TransformerEngine (the NVIDIA-canonical path, opt-in
        # via env subvalue).  Both are gated on the env var; with the FP8
        # env unset only the BF16/FP16 flash path and the naive floor run.
        if _FP8_ENABLED and _FP8_ENV not in ("te", "transformer_engine", "scaled_mm"):
            candidates.append(FP8HopperTritonAttention())
        if _FP8_ENABLED and _FP8_ENV in ("1", "true", "on", "te", "transformer_engine"):
            # TE is added either when explicitly requested or as a fallback
            # under the generic "on" setting.  Never on the AMD-only
            # "scaled_mm" subvalue, never when the user explicitly pinned
            # "triton".
            candidates.append(TransformerEngineAttention())
        # The flash path (FA-3 preferred, FA-2 fallback).  Always present so
        # BF16/FP16 calls land here regardless of FP8 env state.
        candidates.append(HopperFlashAttention())
    elif arch.vendor is Vendor.INTEL:
        # Intel CPU branch.  AMX-aware ops are env-gated for the same reason
        # the GPU FP8 ops are: the perf win is shape-dependent and we don't
        # want to surprise a smoke run.  Candidate order matters — the first
        # op whose supports() returns True wins, so dtype-specific kernels go
        # before the generic SDPA floor.
        if _AMX_ENABLED and _AMX_ENV == "int8":
            from repercep.attention.amx_int8_flash import AMXInt8FlashAttention

            candidates.append(AMXInt8FlashAttention())
        if _AMX_ENABLED and _AMX_ENV in ("1", "true", "on", "fp16"):
            # FP16 kernel only qualifies on Granite Rapids (amx_fp16 flag).
            # On SPR/EMR it stays disqualified, so listing it in the auto-set
            # is harmless — supports() returns False and the chain advances.
            from repercep.attention.amx_fp16_flash import AMXFP16FlashAttention

            candidates.append(AMXFP16FlashAttention())
        if _AMX_ENABLED and _AMX_ENV not in ("ipex", "int8", "fp16"):
            from repercep.attention.amx_flash import AMXFlashAttention

            candidates.append(AMXFlashAttention())
        if _AMX_ENABLED and _AMX_ENV not in ("amx", "int8", "fp16"):
            from repercep.attention.ipex_flash import IPEXFlashAttention

            candidates.append(IPEXFlashAttention())
        # SDPA on CPU is the floor — always present, correct on every shape,
        # and dispatches to oneDNN's AMX matmul automatically on BF16 inputs.
        candidates.append(AMXSDPAAttention())
    candidates.append(NaiveAttention())

    for op in candidates:
        if op.supports(shape, dtype):
            return op
    return NaiveAttention()
