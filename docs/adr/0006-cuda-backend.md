# ADR-0006 — NVIDIA H100 backend lands as a parallel target

- **Status:** Accepted
- **Date:** 2026-05-24
- **Related:** ADR-0001 (MI300X-first hardware target), ADR-0002 (ROCm
  attention primitive), ADR-0003 (vendor-neutral backend Protocol)
- **Resolves:** the "Revisit if" clause of ADR-0001 — by promoting NVIDIA
  from fast-follow to *parallel* track now, ahead of the workload
  trigger ADR-0001 actually named

## Context

The v0.1 verification campaign closed on 2026-05-24 (Session 13, full
record in `docs/SESSION_13_CLOSE.md` + `docs/BUILD_LOG.md`). The
MI300X-side measurement is complete: 142 s headline at 2.68× the H100
published reference, 5-prompt timing variance characterised, multi-prompt
FVD computed, threshold curve mapped. The Python + Rust + Triton stack
is stable; `mypy --strict` holds across 50 source files; `pytest` is
green at 124 cases.

What remains structurally unsatisfying is `docs/METHODOLOGY.md` §3 —
the "apples-to-apples accounting" section explicitly acknowledges that
**the 2.68× claim is system-vs-system, not stack-vs-stack on the same
silicon.** The H100 number (~380 s) is NVIDIA's published reference
stack (TransformerEngine + Apex + NATTEN + flash-attn-3) on a Hopper
GPU we do not have. Repercep's own stack — diffusers + adaptive cache +
FP8 Triton — has never been measured on H100, because no H100 has been
attached to the project.

That changes today. Host is now 1× NVIDIA H100 SXM5 80GB HBM3 (sm_90,
132 SMs, 18 NVLinks @ 26.6 GB/s, 700W TDP confirming SXM5; board PN
692-2G520-0200-000). ROCm is not present on this host. The
architectural question is whether to:

(a) Defer the NVIDIA port until a workload forces it (ADR-0001's
    original "Revisit if" trigger — a priority inference-cloud or AV-simulation
    workload that is H100/H200-bound).

(b) Land it now, because ADR-0003 deliberately made the cost a weekend
    rather than a quarter, and because closing the methodology
    asymmetry is itself valuable.

We are choosing (b).

## Decision

Land the NVIDIA H100 backend as a **parallel target**, not a
fast-follow. AMD MI300X remains the lead workload. Specifically:

1. Add `repercep.backend.cuda.CUDABackend` satisfying the Backend
   Protocol (ADR-0003). Detect via `torch.version.cuda is not None`.
   Map sm_XX → DeviceArch (sm_90 → Hopper, sm_80 → Ampere, sm_89 →
   Ada). Declare capabilities: FP8 (e4m3fn + e5m2 — IEEE-ish, NOT the
   AMD fnuz variant), flash-attention via flash-attn-3, torch.compile.

2. Add three Hopper attention ops:
   - `repercep.attention.hopper_flash.HopperFlashAttention` wrapping
     `flash_attn_interface.flash_attn_func` (FA-3) with
     `flash_attn.flash_attn_func` (FA-2) fallback.
   - `repercep.attention.fp8_hopper_triton.FP8HopperTritonAttention` +
     `kernels/triton_kernels/fp8_flash_attn_hopper.py` — Hopper FP8
     Triton FA-2, sibling of the gfx942 kernel, with separate autotune
     cache at `~/.cache/repercep/fp8_autotune_hopper.json`.
   - `repercep.attention.transformer_engine.TransformerEngineAttention`
     wrapping `transformer_engine.pytorch.DotProductAttention` (FA-3 +
     FP8 recipe) — optional, registers only when TE imports cleanly.

3. `repercep.attention.registry.select_attention_op` grows an NVIDIA
   vendor branch: TE (if available + FP8 env) → FP8 Hopper Triton (if
   env + shape) → FA-3/FA-2 → naive SDPA.

4. `repercep.backend.registry._ALL_BACKENDS` gains `CUDABackend()` as the
   second entry. When both AMD and NVIDIA are visible on the same host
   (rare; CI machines, dev boxes with eGPU) AMD wins — preserves the
   "MI300X is lead" framing under `select_backend()` with no `prefer`
   pin. Single-vendor hosts get whichever is present.

5. New `[nvidia]` optional dependency group in `pyproject.toml`:
   `flash-attn>=3.0` (mandatory for the FA-3 op to register),
   `transformer-engine[pytorch]` (optional). `torch` must be installed
   from the cu128 wheel index, not rocm7.2. `Makefile` notes the wheel
   index choice.

6. `scripts/check_gpu.py` becomes vendor-neutral (prints whichever
   backend is detected; previously ROCm-only).

7. Tests: `tests/test_backend_cuda.py` (sibling of `test_backend.py`) for
   Protocol conformance, identity, capability shape, sm_XX → DeviceArch;
   `tests/test_attention_cuda.py` (sibling of `test_attention.py`) for
   op selection + skip-on-no-CUDA + structural conformance of each new
   op. The FP8-Hopper kernel inherits the same autotune-cache tests the
   gfx942 kernel has — the cache files are separate
   (`fp8_autotune_hopper.json` vs `fp8_autotune.json`) so existing
   `tests/test_attention_fp8.py` cases run unchanged on AMD and skip on
   NVIDIA via a new `torch.version.hip` gate (see F25).

## Rationale

1. **ADR-0003 was deliberately designed for exactly this moment.** The
   Backend Protocol is a single seam; everything above it imports
   `Backend`, not `ROCmBackend`. The cost of the port is **one class +
   one registry entry + the attention ops below the seam** — no
   changes to model loading, the denoise loop, the Cosmos engine, the
   serving handlers, the Rust crates, or the benchmark harness. We
   pay for the ADR-0003 indirection every day; this is the day it
   refunds.

2. **It closes a real asymmetry in METHODOLOGY.md §3.** The published-
   Repercep-MI300X vs published-NVIDIA-H100 framing is *defensible* but
   not the cleanest possible test. With Repercep running on H100, the
   comparison becomes Repercep-MI300X vs Repercep-H100, which is the clean
   apples-to-apples the methodology doc has been honest about lacking.
   That measurement (Session 15) is more valuable to the project's
   credibility than a third decimal place on the MI300X number.

3. **Optionality on the NVIDIA inference-cloud market.** The
   Implementation Plan §5.4 names that market as "subsidized" (NIM,
   TensorRT-LLM); we are not trying to compete head-on there. But
   shipping with NVIDIA support removes a "this is just an AMD
   project" perception and lets the same codebase serve a customer who
   happens to be CUDA-bound for non-cost reasons (existing tooling,
   procurement, etc.). The cost of carrying the optionality is the
   small CI matrix expansion described in *Consequences*; the upside
   is one fewer reason for a prospect to bounce.

4. **The kernel work is the only non-trivial engineering, and it is
   perf-relevant work that does not touch the architecture.** The FP8
   Hopper Triton kernel and the FA-3 wiring are real engineering —
   different SMEM budget (228 KiB/block on Hopper vs 64 KiB LDS on
   gfx942), different MFMA semantics, different FP8 format. But none
   of this touches `Backend`, `WorldModelEngine`, or the denoise loop.
   It sits below the seam where ADR-0003 said it would.

## Consequences

- **AMD remains the lead.** The Cosmos / Wan benchmark headlines, the
  `BUILD_LOG.md` chronology, and the OSS writeup at
  `docs/COSMOS_ON_MI300X.md` — none of these change. The H100 port adds a sibling, not a
  replacement. The forthcoming `docs/COSMOS_ON_H100.md` (template
  written Session 14, numbers pending Session 15) is symmetric to the
  MI300X writeup, not superseding.

- **`select_backend()` order when both vendors visible: AMD first.**
  Preserves the MI300X-lead framing for the default case and for any
  user not explicitly pinning a vendor. Single-vendor hosts get
  whichever is present; multi-vendor hosts can pin with
  `select_backend(prefer="cuda")`.

- **New `[nvidia]` optional dependency group.** `flash-attn>=3.0` is
  *mandatory* in this group — without it `HopperFlashAttention.available`
  is False and only the naive SDPA floor is reachable. `transformer-engine`
  is *optional* (heavy install: cuDNN, custom CUDA toolchain, per-SM C++
  extensions); the Triton FP8 kernel is the path that works without TE.
  TE is the path for users who already have it and gets you FA-3 + FP8
  recipes for free.

- **`torch` must be installed from the cu128 wheel index**, not the
  rocm7.2 one. The Makefile carries a note; mixing the wheel indexes
  produces a torch that supports neither vendor properly.

- **The two FP8 kernels are siblings, not a single parameterized
  kernel.** `tl.float8e4nv` on Hopper (IEEE-ish e4m3fn, FP8_MAX=448,
  inf/nan representable) is a different physical format from
  `tl.float8e4b8` on gfx942 (e4m3 *fnuz*, FP8_MAX=240, no inf/nan).
  The numerical ranges differ by ~2×; the autotune sweet spots
  differ accordingly (Hopper's larger SMEM budget admits 256×256 tiles
  the gfx942 LDS can't hold). Forcing a single parameterized kernel
  would obscure both. They share the FA-2 algorithm and the
  persistent-JSON-cache pattern; the cache files are separated
  (`fp8_autotune.json` vs `fp8_autotune_hopper.json`) so a host that
  later sees both never confuses the two.

- **No multi-GPU support on this host.** The H100 SXM5 here has 18
  NVLinks but no peer to talk to — they are inert. Multi-GPU is Phase
  5+ scope on both vendors; this ADR does not change that.

- **CI matrix grows.** Today CI runs on the MI300X host. The NVIDIA
  path adds a CUDA-host job (most cheaply: a small ephemeral H100 box
  for the test suite; the heavyweight benchmark sweep stays on the
  attached H100 between sessions). Cost is real but bounded — the
  Protocol-conformance tests run in seconds; only the FP8 kernel
  autotune is expensive, and that runs once per host.

- **Documentation surface roughly doubles for the GPU-path docs.**
  `docs/COSMOS_ON_H100.md` mirrors `docs/COSMOS_ON_MI300X.md`. We
  accept the duplication; the alternative (one merged
  `COSMOS_ON_GPU.md` with vendor branches per section) would obscure
  the per-vendor framing and force the reader to disentangle which
  numbers belong to which silicon. Two files, one structure.

## Revisit if

- **The FP8 Hopper Triton kernel autotune does not reach FA-3 + FP8
  (TE) perf at the Cosmos production shape.** In that case TE becomes
  the default Hopper FP8 path and Triton is retained as a fallback for
  installs without TE. The Triton kernel is not load-bearing for the
  port — FA-3 alone is competitive with TE in BF16; FP8 is the
  optimization above that, and either path can deliver it.

- **Multi-GPU lands as a real requirement** (a customer needs 8× H100
  with NVLink-aware scheduling, or 8× MI300X with xGMI, for a model
  Repercep actually serves). That is sharded-attention + topology-aware
  scheduler work; it is structural and beyond the Backend Protocol.
  Plan-residual Phase 5+ on both vendors.

- **An NVIDIA-specific workload pulls scope** (e.g.
  wants TensorRT-LLM interop, NIM packaging, Triton Inference Server
  integration). Those are NVIDIA-ecosystem investments that ADR-0001
  was explicitly skeptical of; if one of them becomes the right
  business decision, this ADR is the place that records the pivot,
  not a quiet refactor.
