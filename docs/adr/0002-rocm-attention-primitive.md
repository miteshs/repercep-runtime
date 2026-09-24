# ADR-0002 — Attention primitive on MI300X

- **Status:** Accepted
- **Date:** 2026-05-22
- **Related:** ADR-0001 (MI300X-first)

## Context

The Implementation Plan (§1.2, §2.2) names **FlashAttention-3** and **NATTEN**
as the attention primitives to build on — "do not write custom CUDA from
scratch." That guidance assumed an H100 lead target.

On MI300X (`gfx942`, CDNA3) neither primitive applies as-is:

- **FlashAttention-3** kernels are Hopper-specific. They are built around
  `wgmma` asynchronous warpgroup matmul and TMA tensor-memory-accelerator
  copies. CDNA3 has neither. FA-3 will not compile or run on gfx942.
- **NATTEN** ships CUDA kernels for neighborhood attention. ROCm/HIP support is
  partial-to-absent depending on version; it cannot be assumed for v0.1.

## Decision

1. Define attention as a Protocol — `AttentionOp` — so the *choice* of kernel is
   a backend-internal detail invisible to model and runtime code.
2. Ship two ops now:
   - `NaiveAttention` (`naive-sdpa`) — `torch.nn.functional.scaled_dot_product_attention`.
     Correct on every shape and dtype. On ROCm, SDPA itself dispatches to
     aotriton-compiled flash kernels when the shape qualifies, so this is a
     respectable floor, not a strawman.
   - `ROCmFlashAttention` (`rocm-ck-flash`) — wraps the AMD Composable-Kernel
     `flash-attn` build, the MI300X-equivalent of FA-3. Currently a thin,
     honest wrapper: if the CK package is absent it advertises
     `available == False` and selection falls back to the floor.
3. Replace NATTEN neighborhood attention with, in order of preference:
   a Triton neighborhood kernel (Triton has a working AMD backend), else the
   naive floor. Tracked as future work, not v0.1.

## Rationale

- The `AttentionOp` Protocol means swapping FA-3 → CK flash-attention touches
  *one backend file*, not model code. This is the abstraction earning its keep.
- Selection is **by problem shape** (`select_attention_op`): a model can use the
  CK flash kernel for its DiT blocks and the naive floor for an unusual
  `head_dim` with no caller-side branching.
- Building on CK rather than hand-writing HIP kernels keeps faith with the
  plan's "use existing primitives" principle — CK *is* the AMD-native existing
  primitive.

## Consequences

- v0.1 may run partly on the naive floor until the CK `flash-attn` package is
  built for gfx942. That is acceptable: the benchmark harness (Task #8) measures
  against the floor, so any CK speedup is captured as real, attributable gain.
- FP8 attention (CDNA3 has native FP8 MFMA) is deferred to the CK-wiring task
  (Task #6), consistent with the plan deferring FP8 to Phase 2.
- A future NVIDIA backend can register a real FlashAttention-3 op under the same
  Protocol with zero changes above the backend layer.
