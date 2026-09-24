# Repercep — Targets and Kernels

**Last updated:** 2026-05-25 (post-Session-15 merge to `main`)

Companion to `docs/architecture.md` (general component map) and the per-vendor
ADRs (`adr/0001-mi300x-first-hardware-target.md`,
`adr/0006-cuda-backend.md`, `adr/0007-cpu-backend.md`).  This doc is the
per-target cut: which silicon, which kernels, which kernel languages, which
toolchain.  Read this when you want to know *what runs on what*.

## Stack overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  USER SURFACES                                                                   │
│                                                                                  │
│   scripts/run_cosmos.py      scripts/run_wan.py      serving/app.py (FastAPI)    │
│   scripts/bench_*.py         scripts/verify_*.py     serving/proto/*.proto (gRPC)│
│   scripts/profile_cosmos.py  repercep.cli (info)       v2: /generate/stream NDJSON │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────────────────────────────────────────────────┐
│  CONTROL PLANE  (Rust + PyO3 crates, maturin-installed into .venv)               │
│                                                                                  │
│   crates/repercep-cache       PagedLatentCache (frame-aware eviction)    [Rust]    │
│   crates/repercep-scheduler   priority + admission control               [Rust]    │
│   crates/repercep-router      request routing                            [Rust]    │
│   src/repercep/runtime/types.py   Frame, GenerationRequest (pydantic)    [Python]  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────────────────────────────────────────────────┐
│  MODEL ENGINES                                                                   │
│                                                                                  │
│   repercep.models.cosmos.CosmosEngine     ── diffusers CosmosTextToWorldPipeline   │
│   repercep.models.wan.WanEngine           ── diffusers WanPipeline (A14B)          │
│   repercep.runtime.engine.WorldModelEngine (Protocol)                              │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────────────────────────────────────────────────┐
│  NATIVE DENOISE LOOP  (Repercep's lever vs the reference)                          │
│                                                                                  │
│   repercep.runtime.denoise.denoise_cosmos_video        [Python]                    │
│     • CFG batching   (one batch=2 forward instead of two sequential)             │
│     • Adaptive cache (TeaCache-style, threshold=0.30) ── skips DiT forward       │
│       when input similarity to prior step is above threshold                     │
│     • Fixed step-skip (--cache-skip-every N, fallback)                           │
│     • `torch.inference_mode()` wrapping (F18 OOM fix)                            │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ATTENTION DISPATCH  (the wedge below SDPA)                                      │
│                                                                                  │
│   repercep.attention.diffusers_backend       ── registers "repercep_fp8"             │
│   repercep.attention.diffusers_backend_amx   ── registers "repercep_amx"             │
│     ↓ both register with diffusers._AttentionBackendRegistry, replacing          │
│       diffusers' default SDPA dispatch when the env var is set.                  │
│                                                                                  │
│   repercep.attention.registry.select_attention_op(arch, shape, dtype)              │
│     ↓ vendor-branched candidate list, env-gated FP8/AMX promotion,               │
│       falls back to NaiveAttention (SDPA) if no specialised op qualifies         │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌────────────────────┐      ┌────────────────────┐      ┌────────────────────────┐
│  TARGET 1: AMD     │      │  TARGET 2: NVIDIA  │      │  TARGET 3: Intel CPU   │
│  Vendor.AMD        │      │  Vendor.NVIDIA     │      │  Vendor.INTEL          │
│                    │      │                    │      │                        │
│  MI300X            │      │  H100 SXM5 sm_90a  │      │  Sapphire Rapids+      │
│  gfx942 / CDNA3    │      │  Hopper / WGMMA    │      │  AMX_BF16 / AMX_TILE   │
│  ROCm 7.2          │      │  CUDA 12.8 cu128   │      │  oneDNN auto-dispatch  │
│  192 GiB HBM3      │      │  80 GiB HBM3       │      │  2x 52c + 2 NUMA       │
│  ROCmBackend       │      │  CUDABackend       │      │  CPUBackend            │
│  (backend/rocm.py) │      │  (backend/cuda.py) │      │  (backend/cpu.py)      │
└────────────────────┘      └────────────────────┘      └────────────────────────┘
```

## Attention kernels per target

### Target 1 — AMD MI300X (`Vendor.AMD`, `gfx942`, ROCm 7.2)

**Production path (default):**
- `NaiveAttention` → `torch.nn.functional.sdpa` → aotriton flash
  &nbsp;&nbsp;&nbsp; *kernel language:* Triton (`.py`)
  &nbsp;&nbsp;&nbsp; *binding:* torch dispatcher; the PyTorch ROCm wheel ships aotriton

**Opt-in via `REPERCEP_FP8_ATTENTION=1`:**
- `FP8TritonAttention` → `kernels/triton_kernels/fp8_flash_attn.py`
  &nbsp;&nbsp;&nbsp; *kernel language:* Triton (`.py`)
  &nbsp;&nbsp;&nbsp; Fused FA-2 in FP8 (E4M3) with per-shape autotune cache.
  &nbsp;&nbsp;&nbsp; Cosmos 121f/36 headline kernel; 1.13× over SDPA at production shape.
- `FP8ScaledMMAttention` → `torch._scaled_grouped_mm`
  &nbsp;&nbsp;&nbsp; *kernel language:* PyTorch op (vendor C++)
  &nbsp;&nbsp;&nbsp; Per-head FP8 GEMM via torch; AMD-only fallback.

**Inert wrapper:**
- `ROCmFlashAttention` → CK flash-attn
  &nbsp;&nbsp;&nbsp; *kernel language:* C++/HIP — package not installed today.

**Scaffold (not productionised):**
- `kernels/hip/fp8_attn/v_mfma_f32_16x16x32_fp8_fp8`
  &nbsp;&nbsp;&nbsp; *kernel language:* HIP C++ + pybind11
  &nbsp;&nbsp;&nbsp; Compiles via hipcc.  Operand register layout was
  incomplete until F48 (2026-08-20) — loaded on 16 of 64 lanes and dropped
  the K-group term — and is now correct and pinned by
  `tests/test_mfma_fragment_layout.py`.  Still correctness-only: the B
  operand gather is column-strided, so promoting this to a perf path needs
  an LDS staging rewrite.  No caller today.

### Target 2 — NVIDIA H100 (`Vendor.NVIDIA`, sm_90a, CUDA 12.8)

**Production path (current default on Hopper — 138.4 s headline):**
- `NaiveAttention` → `torch.nn.functional.sdpa` → cuDNN FA-3
  &nbsp;&nbsp;&nbsp; *kernel language:* cuDNN (vendor CUDA C++)
  &nbsp;&nbsp;&nbsp; torch 2.8+ dispatches BF16 on sm_90 to cuDNN-FA3 automatically.

**Opt-in via `REPERCEP_FP8_ATTENTION`:**
- `FP8HopperTritonAttention` → `kernels/triton_kernels/fp8_flash_attn_hopper.py`
  &nbsp;&nbsp;&nbsp; *kernel language:* Triton (`.py`), WGMMA tiles
  &nbsp;&nbsp;&nbsp; Hopper sibling of the CDNA3 kernel; autotunes over
  `(BLOCK_M, BLOCK_N, num_warps, num_stages={2,3,4,5})`.
  &nbsp;&nbsp;&nbsp; **Status:** 1.7–3.6× slower than cuDNN-FA3 on synthetic Cosmos
  shapes (S=2k–16k); gap narrows at large S.  F29 grid trim landed in the
  post-merge commit (drops `BLOCK_M=192` and `num_warps=12`, both
  power-of-2-required by Triton).  Optimization candidate.
- `TransformerEngineAttention` → `transformer_engine.pytorch.DotProductAttention`
  &nbsp;&nbsp;&nbsp; *kernel language:* TE/CUDA C++ (closed-source kernels)
  &nbsp;&nbsp;&nbsp; F28 install gate: cu13 wheel poisons the resolver; `--no-deps`
  with cu12-only fix documented in `SESSION_15_CLOSE.md`.

**Flash-attn wrapper:**
- `HopperFlashAttention` → `flash_attn_interface` (FA-3) or `flash_attn` (FA-2)
  or SDPA fallback
  &nbsp;&nbsp;&nbsp; *kernel language:* FA-3 = CUTLASS C++; FA-2 = CUDA C++
  &nbsp;&nbsp;&nbsp; FA-3 wheel requires source build (~10 min minimal, ~60 min full).

### Target 3 — Intel CPU (`Vendor.INTEL`, Sapphire Rapids+)

**Production path (always-available floor):**
- `AMXSDPAAttention` → `torch.nn.functional.sdpa` → oneDNN AMX BF16
  &nbsp;&nbsp;&nbsp; *kernel language:* oneDNN (vendor C++ with intrinsics)
  &nbsp;&nbsp;&nbsp; BF16 on SPR+ auto-dispatches to AMX_BF16 `TDPBF16PS`
  instructions through oneDNN's brgemm primitive.

**Opt-in via `REPERCEP_AMX_ATTENTION=1`:**
- `AMXFlashAttention` → `kernels/cpu/amx_attn/flash_attn_amx.cpp`
  &nbsp;&nbsp;&nbsp; *kernel language:* C++17 + `<immintrin.h>` intrinsics + OpenMP
  &nbsp;&nbsp;&nbsp; **Toolchain:**
  - AMX intrinsics: `_tile_loadd`, `_tile_dpbf16ps`, `_tile_stored`,
    `_tile_zero`, `_tile_loadconfig`, `_tile_release`
  - AVX-512 intrinsics: `_mm512_cvtne2ps_pbh` (BF16 cast), `_mm512_fmadd_ps`,
    `_mm512_reduce_max_ps`, polynomial `exp_ps` for softmax
  - OpenMP for outer `(B, H, q_tile)` parallelism
  - Linux `arch_prctl(ARCH_REQ_XCOMP_PERM)` one-shot for AMX tile state
  - Build: `torch.utils.cpp_extension.BuildExtension` via gcc-13 with
    `-march=sapphirerapids -mamx-bf16 -mamx-tile -mavx512bf16`
  - Wrapper: `src/repercep/attention/amx_flash.py`
  &nbsp;&nbsp;&nbsp; **Status (2026-05-25):** built ✓, correct ✓ (0.18 % rel err vs
  SDPA), wins 3.8× at S=1024 / loses 0.5–0.7× at S≥2048 (oneDNN tuning gap —
  optimization candidate).
- `IPEXFlashAttention` → `intel_extension_for_pytorch.ops.fused_*`
  &nbsp;&nbsp;&nbsp; *kernel language:* IPEX / oneDNN (vendor C++)
  &nbsp;&nbsp;&nbsp; Only fires if IPEX is installed.

**INT8 (no FP8 ISA on any shipped Xeon):**
- `runtime/quantize.py` — per-channel symmetric INT8 weight quantization helpers
- The AMX INT8 attention kernel itself is deferred to Session 16+.

## Headline numbers per target

| Target | Wall time | Config | vs NVIDIA H100 reference (~380 s) |
|---|---:|---|---:|
| MI300X | **142.0 s** | Cosmos 121f/36, adaptive cache + autotuned FP8 Triton | **2.68×** |
| H100 (FA-3 path, 2026-05-25) | **99.6 ± 3.9 s** (5-prompt mean ± std; min 95.2 s) | Cosmos 121f/36, adaptive cache + FA-3 via bridge | **3.81×** mean / 3.99× best |
| H100 (FA-2 path) | **132.5 s** | Cosmos 121f/36, adaptive cache + FA-2 via bridge | 2.87× |
| H100 (torch SDPA, prior baseline) | **138.4 s** | Cosmos 121f/36, adaptive cache alone | 2.75× |
| H100 (FP8 Triton, current) | **343.7 s** | Cosmos 121f/36, adaptive + Repercep FP8 kernel | 1.10× (FP8 kernel needs surgery) |
| Intel CPU SPR | **875.6 s** | Cosmos 17f/8, AMX-aware path, 48 threads | substrate |

The 95.3 s headline requires FA-3 built from source (the PyPI flash-attn
wheel ships FA-2 only) and `REPERCEP_FP8_ATTENTION=fa` set to activate the
bridge:

```bash
# One-time: build FA-3 minimal for sm_90 (~5 min)
cd /tmp && git clone --depth 1 https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper && \
  FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
  FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
  FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE \
  FLASH_ATTENTION_DISABLE_FP8=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
  FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
  MAX_JOBS=8 uv pip install --python /path/to/.venv --no-build-isolation .

# Headline reproducer
REPERCEP_FP8_ATTENTION=fa .venv/bin/python scripts/run_cosmos.py --backend cuda \
    --frames 121 --steps 36 --native-loop \
    --cache-mode adaptive --cache-adaptive-threshold 0.30 \
    --cache-force-full-every 16
# expect: generate_seconds ≈ 95 s
```

CPU per-step extrapolation to 121 f / 36 steps: ~430 s/step × 36 ≈ 4–5 hours
without caching / kernel optimization.  See `docs/COSMOS_ON_CPU.md` for the
optimization plan.

### Why FA-3 beats cuDNN-via-SDPA by 2× on Hopper

`bench_fp8_hopper.py` measured the gap at the Cosmos shapes:

| seq_len | torch SDPA (cuDNN) | HopperFlashAttention (FA-3) | speedup |
|--------:|------------------:|---------------------------:|--------:|
| 8 192   | 6.06 ms           | 2.95 ms                    | 2.05×   |
| 16 384  | 23.94 ms          | 13.14 ms                   | 1.82×   |
| 32 768  | 101.23 ms         | 51.43 ms                   | 1.97×   |

torch 2.8.0+cu128's SDPA dispatches BF16 on sm_90 to cuDNN, but the
cuDNN path it chooses is not the same WGMMA-based FA-3 kernel that the
Dao-AILab wheel ships.  The standalone FA-3 wheel runs the
hand-tuned-for-Hopper kernel from the FA-3 paper directly, which
explains the consistent 1.8–2.0× margin and the 31 % end-to-end win.

## Language / toolchain summary

| Layer | AMD MI300X | NVIDIA H100 | Intel CPU |
|---|---|---|---|
| Runtime orchestration | Python | Python | Python |
| Control plane | Rust + PyO3 | Rust + PyO3 | Rust + PyO3 |
| Production attention | Triton (aotriton via SDPA) | C++ (cuDNN-FA3 via SDPA) | C++ (oneDNN via SDPA) |
| Own fused attention | Triton (`.py`) | Triton (`.py`) | **C++ with AMX/AVX-512 intrinsics + OpenMP** |
| FP8 / low-precision | FP8 E4M3 (Triton), `torch._scaled_grouped_mm` | FP8 E4M3 (Triton WGMMA), TransformerEngine | INT8 (PyTorch); no FP8 ISA |
| Optional vendor SDK | (CK flash-attn — inert) | flash-attn FA-3 (CUTLASS), TransformerEngine | IPEX |
| Scaffold | HIP C++ (`kernels/hip/fp8_attn`) | — | — |
| Build tooling | `make rust-install` (maturin), Triton JIT, ROCm wheel | maturin, Triton JIT, CUDA wheel, optional source FA-3 | maturin, `make kernels-cpu` (torch CppExtension + gcc) |

## Key architectural property

Every concrete vendor lives *under* one Protocol (`Backend`) and one selector
(`select_attention_op`) — adding a new vendor is N new files in `backend/` +
`attention/` plus an entry in the `_ALL_BACKENDS` tuple.  The denoise loop,
the model engines, serving, and the Rust control plane don't know which vendor
is active.

That's why the same invocation:

```bash
scripts/run_cosmos.py --backend cpu --frames 17 --steps 8
```

produced the 875.6 s CPU number on 2026-05-25 with zero changes to anything
above the attention dispatcher.
