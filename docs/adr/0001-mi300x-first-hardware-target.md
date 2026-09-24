# ADR-0001 — MI300X is the lead hardware target

- **Status:** Accepted
- **Date:** 2026-05-22
- **Supersedes:** the Implementation Plan's Phase-1 hardware choice (H100-first)

## Context

The Repercep Implementation Plan (§1.2, §1.4 Decision 3) sequences hardware as:
H100 first, H200 fast-follow, and MI300X *or* Jetson Thor as a Phase-5 second
target around Month 12–18. The reasoning was that NVIDIA documents Cosmos
inference characteristics in detail, which de-risks the baseline measurement.

We are deliberately inverting this: **MI300X is Phase-1.**

## Decision

Build the first runnable Cosmos-Predict-7B path on AMD Instinct MI300X
(`gfx942`, CDNA3) on ROCm 7.2. Treat NVIDIA as a fast-follow, kept cheap by the
vendor-neutral backend Protocol (ADR-0003).

## Rationale

1. **The plan's own competitive analysis points here.** §5.4 Principle 3 and
   §7.1 state the defensible wedge is *non-NVIDIA silicon*: "On NVIDIA silicon,
   NIM and TensorRT-LLM are subsidized. On AMD, Trainium, Apple Silicon ... there
   is no production-grade WM serving today." Leading on H100 means launching
   into a market with a free, bundled, vertically-integrated incumbent. Leading
   on MI300X means launching where the value proposition is "this is the only
   production-grade option that exists." The H100-first plan and the
   non-NVIDIA-wedge thesis were in tension; this resolves it.

2. **The hardware is in hand now.** Development is gated on an available MI300X
   (192 GiB HBM3, 304 CUs, gfx942). No procurement delay; no Phase-5 wait.

3. **192 GiB HBM removes memory pressure from v0.1.** Cosmos-Predict-7B in bf16
   is ~14 GB of weights. On an 80 GB H100, latent cache + activations + text
   encoder force memory engineering early. On MI300X the whole model plus a
   generous latent cache fits with room to spare, so v0.1 can focus on the
   correctness and attention path rather than memory tetris.

4. **It is reversible at near-zero cost.** ADR-0003's Protocol-based backend
   means an NVIDIA backend is an *added class*, not a refactor. We lose nothing
   by starting on AMD.

## Consequences

- **FlashAttention-3 is off the table as the primary kernel.** FA-3 is
  Hopper-specific (`wgmma`, TMA). The MI300X path uses Composable-Kernel
  flash-attention / aotriton SDPA instead — see ADR-0002.
- **No NVIDIA-documented baseline.** The plan leaned on NVIDIA's published
  Cosmos inference numbers. Our baseline is instead PyTorch + Diffusers running
  on ROCm, measured by our own harness (Task #8). This is arguably more honest —
  the benchmark paper compares like-for-like on the same silicon.
- **ROCm toolchain maturity is a risk.** ROCm 7.2 is recent; some kernels
  (notably NATTEN neighborhood attention) have weak or no ROCm support. The
  `AttentionOp` Protocol contains this risk — unsupported ops fall back to the
  naive SDPA floor (ADR-0002).
- **Strategic upside:** a working Cosmos-Predict-7B serving path on MI300X is,
  per the plan, something *no production-grade stack currently provides*. That
  is a sharper Phase-1 differentiator than a 2–3x speedup on H100 where NIM
  already exists.

## Revisit if

A priority inference-cloud or AV-simulation workload turns out to be
H100/H200-bound and needs parity soon — at which point the NVIDIA backend is
promoted from fast-follow to parallel track.
