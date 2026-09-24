# ADR-0007 — CPU backend (Sapphire Rapids AMX-first)

**Status:** Accepted (2026-05-24)
**Context:** ADR-0001 (MI300X-first), ADR-0003 (vendor-neutral Backend Protocol),
ADR-0006 (NVIDIA H100 parallel target).

## Decision

Land a third compute backend, `repercep.backend.cpu.CPUBackend`, that satisfies
the same `Backend` Protocol as `ROCmBackend` and `CUDABackend`.  Lead target
is **Intel Sapphire Rapids** (`spr`, family 6 model 143) because it is the
first widely-deployed x86 silicon with AMX matrix-multiply instructions
(`AMX_BF16` for BF16, `AMX_INT8` for INT8) and because it is the host CPU on
both the AMD MI300X and the NVIDIA H100 RunPod templates Repercep develops
against — every developer who runs Repercep on a GPU also has Sapphire Rapids
sitting underneath.

The three Intel uarchs the registry knows about — Sapphire Rapids,
Emerald Rapids, Granite Rapids — share the AMX_BF16 + AMX_INT8 ISA;
Granite Rapids adds AMX_FP16 + AMX_COMPLEX which we do not yet wire.  A
fourth bucket `generic-avx512` covers pre-AMX Xeon (Ice Lake, Cascade
Lake) and a `generic-x86_64` bucket covers AMD EPYC and anything older —
both still produce a `CPUBackend` that runs the SDPA floor; only the AMX
attention ops disqualify themselves.

Three attention ops land alongside the backend, mirroring the per-vendor
sibling pattern from ADR-0006:

| Op | Module | Role |
|---|---|---|
| `AMXFlashAttention` | `repercep/attention/amx_flash.py` | Custom AMX BF16 flash kernel.  Sibling of the gfx942 + Hopper Triton kernels.  C++ source in `kernels/cpu/amx_attn/`.  Wins at long S because it does not materialise QK^T. |
| `IPEXFlashAttention` | `repercep/attention/ipex_flash.py` | Wrapper over Intel-Extension-for-PyTorch's fused attention.  Available only when IPEX is installed.  Broader shape coverage than the AMX flash kernel. |
| `AMXSDPAAttention` | `repercep/attention/amx_sdpa.py` | `torch.nn.functional.scaled_dot_product_attention` on CPU.  Always available; on AMX-class hardware oneDNN's matmul dispatch picks AMX tiles automatically.  The floor. |

Routing is gated on `REPERCEP_AMX_ATTENTION` (parallels `REPERCEP_FP8_ATTENTION`):
unset, SDPA is the only op the INTEL branch considers; set to `1/true/on`,
the AMX flash kernel is preferred when it qualifies; explicit `=amx` or
`=ipex` forces one or the other.

## Why land CPU now

1. **Substrate completeness.**  The Backend Protocol's promise is "one
   class per vendor, no model-side code changes."  Until CPU was a class,
   the promise had a quiet caveat: "GPU vendor."  Landing CPU completes the
   contract — and shakes out latent assumptions in the Protocol that were
   GPU-shaped (e.g. that `total_memory_bytes` meant VRAM).
2. **CI without a GPU.**  Repercep's pytest suite includes 12 vendor-conditional
   tests that today skip on a CPU-only host.  With CPUBackend in place, the
   floor tests run on every host — no more "passes on my GPU box, breaks on
   the laptop" gaps.
3. **Numerical-parity oracle.**  CPU SDPA is the highest-fidelity reference
   for the GPU flash kernels.  When a Triton or aotriton kernel produces
   suspect output, running the same `(Q, K, V)` through `AMXSDPAAttention`
   on CPU is the comparator that tells us whether the bug is in the kernel
   or the model.
4. **Edge-class story.**  The roadmap includes smaller world models
   (sub-2B parameters) where CPU at AMX speeds is *actually* competitive
   with discrete GPUs at edge power budgets.  Today's effort lays the
   substrate; Granite Rapids + AMX_FP16 in 2026 turns it into a real wedge.

## What we explicitly do not promise

* **Cosmos-Predict-7B on CPU at production latency.**  The 7B-parameter
  diffusion-temporal video model is GPU-class.  CPU runs are minutes per
  video at best, even on a 96-core SPR with AMX.  COSMOS_ON_CPU.md
  documents this honestly — the value of the run is that it *runs*, not
  that it's fast.
* **FP8 on CPU.**  No shipped Intel Xeon has FP8 ISA support today.  When
  one ships (post-Granite Rapids), we extend `capabilities()` to advertise
  it; until then the backend never lies about its capabilities.
* **A custom AMX INT8 attention kernel.**  AMX_INT8 (TDPBSSD) is on the
  roadmap as `repercep/attention/amx_int8_flash.py` — sibling of the BF16
  kernel — but is not in scope for this ADR.  Weight quantization will
  land first (per-channel symmetric INT8, via `torchao`) so the SDPA path
  benefits even without the custom kernel.

## Consequences

* The substrate matrix is now `(AMD | NVIDIA | INTEL) × (gfx942 | sm_90a |
  spr | emr | gnr)`.  Adding a new vendor (Apple Metal, Qualcomm Adreno)
  remains a single-Backend-class operation.
* `repercep info` now reports a CPU device on every host, with feature
  flags (AMX/AVX-512) surfaced so the diagnostic answers "is this host
  going to be fast?" without a kernel run.
* The Backend Protocol grows no fields.  CPU squares the "no GPU
  assumptions" claim by being the third concrete implementation that
  fits the Protocol without parameter additions.

## Alternatives considered

* **Skip CPU entirely; rely on `torch.device("cpu")` from inside model
  code.**  Rejected: this leaks the substrate seam into Cosmos/Wan engine
  code, the exact failure mode ADR-0003 was written to prevent.  Every
  future model addition would re-write the same `if device.type == 'cpu'`
  branches.
* **Wrap only IPEX, skip the custom AMX kernel.**  Rejected as too thin a
  win: IPEX is an *optional* install (~1 GiB) and is not packaged for
  every Xeon SKU.  The custom kernel ships in-tree and is the only path
  with no third-party install dependency.
* **Use oneDNN directly via its C API.**  Rejected as overkill for the
  attention surface — oneDNN's primitive abstraction adds a translation
  layer between Repercep's shape types and the AMX tile registers without
  giving us anything that hand-written AMX intrinsics don't.  oneDNN
  remains the path used *underneath* `torch.matmul` for the BF16 cases
  the SDPA floor catches.

## Open work

| Item | Module | Status |
|---|---|---|
| AMX INT8 attention kernel | `repercep/attention/amx_int8_flash.py` + `kernels/cpu/amx_int8_attn/` | **Code complete (Session 18, Item B).**  Build hardware-gated on `amx_int8`. |
| Weight quantization (INT8 symmetric) | `repercep/runtime/quantize.py` | **Done (Session 18, Item A).**  `QuantizedLinear`/`QuantizedLinearModule`/`replace_linears_with_quantized` landed; CPU/CUDA bit-parity validated on real Ada (Item H). |
| AMX_FP16 path for Granite Rapids | `repercep/attention/amx_fp16_flash.py` + `kernels/cpu/amx_fp16_attn/` | **Scaffolded (Session 18, Item C).**  Inner `_tile_dpfp16ps` loop marked `// TODO(GNR):`; awaits real Granite Rapids host. |
| CPU Cosmos runs against the no-cache reference | `docs/COSMOS_ON_CPU.md` | **Hardware-blocked.**  Every dev VM today reports AMX masked; numbers stay TBD until bare-metal Sapphire/Emerald/Granite Rapids access. |
| ipex install pin in `[cpu]` extra | `pyproject.toml` | **Closed (Session 18, Item D).**  PEP 508 marker `sys_platform == 'linux' and platform_machine == 'x86_64'` added so off-platform installs no longer error. |
| Registry + capabilities wiring for INT8/FP16 ops | `repercep/attention/registry.py` + `repercep/backend/cpu.py` | **Done (Session 18, INTEG).**  New `REPERCEP_AMX_ATTENTION=int8`/`=fp16` subvalues; `capabilities()` probes `amx_int8`/`amx_fp16` flags. |
| CPU FVD + LPIPS evaluation runner | `scripts/eval_cpu_quality.py` | **Done (Session 18, Item F).**  CPU/CUDA parity within 1.5e-5 for LPIPS, bit-identical MSE/PSNR (Item I). |
| `--vae-tiling` refused on CPU backend | `scripts/run_wan.py` + `docs/WAN_ON_CPU.md` | **Done (Session 18, Item E).**  Hard exit with reference to Session 17 methodology miss. |
