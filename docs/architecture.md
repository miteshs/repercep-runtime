# Repercep Runtime — Architecture

This document is the component map. Decision rationale lives in `docs/adr/`.

## Design principle

> A senior architect spending the first weeks doing nothing but interface
> design, type hierarchy, and component boundaries is investing in productivity
> that compounds for the next 18 months. — *Repercep Implementation Plan, §3.4*

Repercep is built outside-in: every layer depends only on the **Protocol** of the
layer beneath it, never on a concrete implementation. The non-kernel codebase is
`mypy --strict`. The kernel layer (future) is a separate codebase with different
rules — mixing clean-architecture and perf code yields clean-but-slow or
fast-but-unmaintainable.

## Layer stack

```
        ┌─────────────────────────────────────────────┐
        │  serving/   frame-streaming HTTP + gRPC API  │   (next)
        ├─────────────────────────────────────────────┤
        │  runtime/   inference engine · scheduler ·   │   (next)
        │             paged latent cache               │
        ├─────────────────────────────────────────────┤
        │  models/    Cosmos-Predict-7B: DiT, tokenizer,│  (next)
        │             text encoder, diffusion loop      │
        ├──────────────────────┬──────────────────────┤
        │  attention/          │  backend/             │  ← LANDED
        │  AttentionOp Protocol │  Backend Protocol     │
        │  naive · rocm_flash   │  ROCmBackend (gfx942) │
        ├──────────────────────┴──────────────────────┤
        │  hardware.py   Vendor · DeviceArch ·          │  ← LANDED
        │                DeviceSpec · DType             │
        └─────────────────────────────────────────────┘
                              │
                   PyTorch (ROCm 7.2 / HIP)
                              │
                      AMD Instinct MI300X
```

## The one seam: `Backend`

`repercep.backend.protocol.Backend` is the single seam between Repercep and a GPU
vendor. Everything above it is written against the Protocol only. Adding NVIDIA
support = writing one class that satisfies `Backend`; no existing code changes.
That is how MI300X can lead without painting the project into a corner — see
[ADR-0001](adr/0001-mi300x-first-hardware-target.md) and
[ADR-0003](adr/0003-vendor-neutral-backend-protocol.md).

The backend is *not* a re-abstraction of PyTorch — torch already hides
CUDA-vs-HIP. The backend is the **policy layer above torch**: capability
declaration, device selection, dtype policy, and kernel (attention op) choice.

## Attention

`AttentionOp` (`repercep.attention.protocol`) is the contract every attention
kernel implements. `select_attention_op` picks the fastest op that supports a
given problem shape on a given architecture:

- `NaiveAttention` — torch SDPA. Correct everywhere; the benchmark floor.
- `ROCmFlashAttention` — Composable-Kernel `flash-attn` for gfx942. Replaces the
  plan's Hopper-only FlashAttention-3 — see
  [ADR-0002](adr/0002-rocm-attention-primitive.md).

World models need three patterns (`AttentionKind`): `FULL` (DiT spatial),
`CAUSAL` (temporal autoregression), `NEIGHBORHOOD` (NATTEN-style local).

## What's next (task order)

1. **runtime/** — Pydantic request/response types, inference engine skeleton,
   paged latent cache (frame-aware eviction).
2. **serving/** — frame-level streaming HTTP + gRPC contracts.
3. **attention/** — real CK flash-attention wiring + FP8.
4. **models/** — Cosmos-Predict-7B loader and single-MI300X inference path.
5. **bench/** — harness vs naive PyTorch + Diffusers (measurement before
   optimization).
