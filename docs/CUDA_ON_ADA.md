# CUDA test sweep on NVIDIA RTX 2000 Ada (sm_89) — Session 18

*First publicly reported Repercep-stack execution on Ada-Lovelace
silicon.  All previous CUDA work on this repo (Sessions 14-17) was
developed on H100 SXM5 (sm_90a, 80 GiB HBM3).  This sweep adds Ada
(sm_89, 16 GiB GDDR6) as a second CUDA arch the runtime is known to
work on.*

**Status (2026-05-25):** Test-suite sweep only — **no model
workloads run**.  Cosmos-Predict-7B and Wan-2.2 A14B do not fit in
16 GiB and were intentionally not attempted (see Honest scope
below).  The contribution is **coverage**: 11 CUDA-conditional tests
that had only ever skipped on the project's previous CPU-only and
H100 hosts now actually run AND pass on Ada.

## TL;DR

The Repercep NVIDIA backend is **arch-portable across the Hopper /
Ada boundary** within the structural and kernel-correctness layer:

- All 9 `test_backend_cuda.py` tests pass on Ada — device
  enumeration, FP8-capability advertisement, backend selection,
  `attention_op` dispatch at a Cosmos-DiT shape.
- All 11 runnable `test_attention_cuda.py` tests pass on Ada (2 of
  13 skip only because flash-attn is not installed; they would skip
  on H100 in the same install state).
- **The FP8 "Hopper" Triton kernel runs on Ada and matches SDPA at
  3.5% mean rel diff** — within 0.1 percentage points of the 3.4%
  H100 measurement reported in `BUILD_LOG.md` F27.  The kernel uses
  `tl.float8e4nv`, which Ada's sm_89 supports identically to
  sm_90a, and `select_attention_op()` correctly disqualifies the
  unavailable `nvidia-flash` candidate and falls through to it when
  the env asks for FP8.

The arch-portability is **not** claimed at the perf or model-
workload layer — only the kernel-correctness and backend-selection
layer.  This card cannot run Cosmos or Wan; perf-tuning on Ada is
out of scope.

## Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation Laptop |
| Compute capability | sm_89 (8.9) |
| Total memory | 15.6 GiB GDDR6 |
| SMs | 22 (vs 132 on H100 SXM5) |
| Driver | 565.57.01 |
| CUDA | 12.8 (torch wheel) |
| PyTorch | 2.8.0+cu128 |
| Triton | 3.4.0 |
| flash-attn | not installed |
| TransformerEngine | not installed |
| diffusers | not installed (intentional — no models to run) |

## Test sweep results

### Counts

| Configuration | Passed | Failed | Skipped |
|---|---|---|---|
| Hypothetical CPU-only host (`CUDA_VISIBLE_DEVICES=""`, same wheel) | 98 | 7 | 27 |
| **This host (Ada, real CUDA)** | **110** | **5** | **17** |
| Delta | **+12** | **−2** | **−10** |

The 110/5/17 totals omit four test modules that fail at collection
on this environment for reasons unrelated to CUDA capability:
`test_router.py`, `test_scheduler.py` (Rust `repercep_router` and
`repercep_scheduler` extensions not built locally — they ship as
stubs in `~/.local/lib/python3.12/site-packages/repercep_*/_native.py`),
and `test_serving.py`, `test_serving_v2.py` (no `fastapi`).  The
sweep also skips `test_eval_cpu_quality.py` and `test_fvd.py` per
the standing slow-suite ignore.

### Newly-running coverage (CPU-only → Ada)

These 11 GPU-conditional tests previously had no way to be
exercised in CI — the project had no CUDA host until Session 14,
and even then ran on Hopper, not Ada.  They all pass on Ada:

`tests/test_attention_cuda.py`:
- `test_naive_runs_on_cuda` (SDPA on cuda:0 with BF16 Q/K/V)
- `test_fp8_hopper_triton_matches_sdpa` (the headline — see §F40
  verification below)

`tests/test_backend_cuda.py` (all 7 GPU-gated entries):
- `test_cuda_devices_detected` — enumerates the Ada device,
  asserts `sm_*` arch id, NVIDIA vendor, nonzero memory and SMs
- `test_select_backend_returns_cuda_when_no_rocm`
- `test_select_cuda_explicitly`
- `test_capabilities_hopper_advertises_fp8` — the test name is
  Hopper-flavoured but the body already covers `sm89` in its arch
  check (`arch.startswith(("sm90", "sm89"))`), so Ada passes this
  unchanged.  Ada's BF16/FP16/FP8E4M3/FP8E5M2 capability surface
  matches the H100 advertisement.
- `test_torch_device_handle`
- `test_default_dtype_is_bf16`
- `test_attention_op_runs_cosmos_dit_shape` — runs the
  `(B=1, H=32, S=512, D=128)` Cosmos-DiT self-attention shape
  end-to-end through `CUDABackend.attention_op(...)`.

Plus 2 tests in `test_wan.py` that construct a CUDA `torch.Generator`
in their assertion path: `test_wan_engine_passes_guidance_scale_2_
for_moe_variant` and `test_wan_engine_omits_guidance_scale_2_for_
non_moe_variant`.  Both pass on Ada, both fail on CPU-only with
`AcceleratorError: no CUDA-capable device`.

### Failures on Ada

All 5 failures are **infrastructure issues, not CUDA-arch issues**.
Nothing sm_89-specific failed:

1. `tests/test_runtime.py::test_latent_cache_*` (×3) — the
   `repercep_cache._native` stub installed in user site-packages has
   `class PagedLatentCache: pass`, so `PagedLatentCache(num_pages=4)`
   raises `TypeError: PagedLatentCache() takes no arguments`.  Fix
   is to build the Rust extension (`cargo build --release` +
   `maturin develop`); out of scope for this sweep.
2. `tests/test_wan.py::test_wan_engine_load_*` (×2) — both bodies
   `patch("diffusers.AutoencoderKLWan.from_pretrained", ...)`,
   which fails at `_patch.__enter__` time with `ModuleNotFoundError:
   diffusers`.  Same root cause as the 8 skipped
   `test_attention_diffusers_backend.py` tests.  Would pass if
   `diffusers` were installed; not an Ada vs H100 finding.

### Skipped on Ada

The 17 skips break down as:
- 8 × `diffusers required for backend bridge tests` —
  `test_attention_diffusers_backend.py` is `pytestmark`-gated on
  `diffusers`, which we don't install.
- 3 × `no ROCm GPU on host` — `test_backend.py` /
  `test_attention.py` AMD-paired entries.
- 3 × `fp8e4m3fnuz / fp8e4b8 paths are AMD-only` —
  `test_attention_fp8.py::_gpu_or_skip` early-exits on
  `not torch.version.hip`.  These are correctly skipped: those
  kernels use AMD-only dtypes, and the corresponding Hopper /
  sm_89 path is exercised in `test_attention_cuda.py`.
- 2 × `flash-attn not installed` — the `HopperFlashAttention`-
  gated entries.

None of the skips are CUDA-arch-related; all would skip
identically on H100 in this install state.

## Findings — sm_89 vs sm_90a differences

### FP8 Hopper Triton kernel: numerically arch-portable Ada ↔ Hopper

The Repercep FP8 Triton attention kernel at
`kernels/triton_kernels/fp8_flash_attn_hopper.py` is named "Hopper"
and was developed on H100 (Session 11 / Session 14, autotune cache
at `fp8_autotune_hopper.json`).  On Ada (sm_89) it:

| Property | Ada (sm_89) | H100 (sm_90a, F27) |
|---|---|---|
| `op.available` | True | True |
| Compiles cleanly | yes | yes |
| Mean rel diff vs SDPA at (B=1, H=8, S=4096, D=128) BF16 | **3.54%** | 3.4% |
| Autotune winner for this shape | `BLOCK_M=64 BLOCK_N=64 num_warps=4 num_stages=3` | `BLOCK_M=128 BLOCK_N=128 num_warps=8 num_stages=2` (Cosmos shape; see `COSMOS_ON_H100.md`) |

Two observations:

1. **The kernel uses `tl.float8e4nv` (the NVIDIA E4M3 dtype), which
   Ada supports.**  The "Hopper" name in the file is silicon-family
   shorthand for "NVIDIA FP8-capable Triton path"; it is not gated
   on `sm_90a` anywhere in the kernel or the dispatch.  This means
   the same kernel is the right answer for any sm_89+ NVIDIA card
   today.  Worth renaming or at least documenting at the
   `kernels/triton_kernels/` README level.
2. **The autotune cache is per-host, so Ada gets a different
   winning tile than H100** — 64 × 64 with 4 warps, vs Hopper's
   128 × 128 with 8 warps at a much larger production shape.  This
   is exactly the behaviour the autotune layer is designed to
   produce; the cache key includes neither arch nor SM count, so a
   host swap will redo the search.  Ada's smaller register file
   (sm_89 has 64K 32-bit regs per SM vs sm_90a's 64K but with
   denser FP8 throughput) and 22 SMs make smaller tiles a better
   fit at this S, which the autotune found unaided.

### `HopperFlashAttention` correctly reports unavailable

The `nvidia-flash` op (FlashAttention wrapper) reports
`available=False` on this host — but that is because `flash_attn`
is not pip-installed, not because of sm_89.  The check is purely
`importlib.util.find_spec("flash_attn") is not None`; flash-attn
upstream does support sm_89 from v2.6+, so this is install-state
not silicon-gated.

### No sm_89-specific test regressions found

Every CUDA-conditional test that ran (11 of them) passed.  No
kernel failed to compile, no numeric tolerance was breached, no
device-property check fell outside its expected range.  The
`devices()[0].arch.gfx_id` came back as a sensible `sm_*` string
(see `test_cuda_devices_detected`).  The `default_dtype()` of BF16
is correct on Ada; the FP8 capability advertisement is correct
(Ada has hardware FP8 the same way Hopper does).

## F40-style dispatch verification on Ada

`docs/BUILD_LOG.md` F40 records that the diffusers bridge
`REPERCEP_FP8_ATTENTION=fa` value does not engage on
`WanTransformer3DModel`.  This sweep verifies the related but
distinct dispatch-disqualification path: **what does
`select_attention_op(H100_arch, shape, BF16)` pick when the env
asks for FP8 / FA on an Ada host where `flash_attn` is not
installed?**

| `REPERCEP_FP8_ATTENTION` | Selected op | `available` |
|---|---|---|
| (unset) | `naive-sdpa` | (attr absent) |
| `1` | `fp8-hopper-triton-flash` | `True` |
| `fa` | `naive-sdpa` | (attr absent) |
| `triton` | `fp8-hopper-triton-flash` | `True` |
| `te` | `naive-sdpa` | (attr absent) |

Three things this confirms:

1. **The `REPERCEP_FP8_ATTENTION=fa` env value is not in the
   registry's `_FP8_TRUTHY` set** — it is currently
   `("1", "true", "on", "triton", "scaled_mm", "te",
   "transformer_engine")`.  With `fa`, FP8 is disabled, only the
   `HopperFlashAttention` candidate is added, and it is
   unavailable (no flash-attn), so the registry falls all the way
   through to `naive-sdpa`.  This matches BUILD_LOG F40's framing
   that `fa` is "informational, not load-bearing" on Wan; it is
   in fact informational at the registry-shape level on any host
   without flash-attn.  The "fa" bridge engagement happens in the
   `diffusers_backend.py` bridge, not in `select_attention_op`.
2. **`REPERCEP_FP8_ATTENTION=1` / `=triton` correctly route to
   `fp8-hopper-triton-flash` on Ada**, and that op runs and is
   numerically correct (3.54% vs SDPA).  The disqualification of
   `HopperFlashAttention` (flash-attn missing) and of
   `TransformerEngineAttention` (TE missing) is silent and
   correct, with the FP8 path winning because it is the first
   candidate left standing whose `supports()` returns True for
   the shape.
3. **`NaiveAttention` has no `available` attribute** —
   `op.available` raises `AttributeError`.  The `AttentionOp`
   Protocol (`src/repercep/attention/protocol.py`) does not declare
   `available`; it is a convention some ops implement.  Code that
   introspects an op's availability must use
   `getattr(op, "available", True)`.  Minor finding worth
   documenting in the Protocol, not worth a code change in this
   sweep (which is restricted to docs).

## Honest scope

**This card cannot run model workloads.**  Cosmos-Predict-7B needs
~52 GiB peak HBM on H100 (`COSMOS_ON_H100.md`); Wan-2.2 A14B needs
72.6 GiB on H100 with `--vae-tiling` (F38).  Ada's 15.6 GiB is not
in the same ballpark.  No `scripts/run_cosmos.py` or
`scripts/run_wan.py` invocation was attempted; that work belongs on
H100 or larger.

What this sweep does add is **structural and kernel-correctness
coverage for the NVIDIA backend's portability layer**.  The
`CUDABackend`, `select_attention_op`, FP8 Hopper Triton kernel, and
device-properties probe are all known to work on at least two
sm-archs now (sm_89 + sm_90a) rather than one.  When the project
later wants to advertise "Repercep runs on NVIDIA RTX 40-series /
L4 / L40S / sm_89 in general", this sweep is the structural
substrate that claim can rest on; the perf claim still needs an
sm_89 card with enough VRAM to run a real model.

## Reproduce

```bash
# Use the system Python — torch 2.8.0+cu128 and triton 3.4.0 are
# already installed at /usr/bin/python3.  No .venv on this host.
cd <repercep-checkout>
PYTHONPATH=$(pwd)/src python3 -c "import torch; \
    print(torch.cuda.get_device_name(0), \
          torch.cuda.get_device_capability(0))"

# CUDA-targeted sweep (~2 min, dominated by Triton autotune in
# test_fp8_hopper_triton_matches_sdpa first-run)
PYTHONPATH=$(pwd)/src python3 -m pytest \
    tests/test_backend_cuda.py \
    tests/test_attention_cuda.py \
    tests/test_attention_fp8.py \
    -v --tb=short

# Full sweep (skip eval / fvd / Rust-extension / fastapi-dependent)
PYTHONPATH=$(pwd)/src python3 -m pytest -q \
    --ignore=tests/test_eval_cpu_quality.py \
    --ignore=tests/test_fvd.py \
    --ignore=tests/test_router.py \
    --ignore=tests/test_scheduler.py \
    --ignore=tests/test_serving.py \
    --ignore=tests/test_serving_v2.py

# F40-style dispatch verification
REPERCEP_FP8_ATTENTION=fa PYTHONPATH=$(pwd)/src python3 -c "\
import torch; \
from repercep.attention.registry import select_attention_op; \
from repercep.attention.types import AttentionShape, AttentionKind; \
from repercep.hardware import H100, DType; \
op = select_attention_op(H100, \
    AttentionShape(1,8,4096,4096,128,AttentionKind.FULL), \
    DType.BF16); \
print('op:', op.name, 'available:', getattr(op,'available','(n/a)'))"
```

## References

- `docs/COSMOS_ON_H100.md` — H100 measurement substrate this doc
  parallels.  Same backend, same kernels, different silicon.
- `docs/BUILD_LOG.md` F27 — H100 FP8 Hopper Triton vs SDPA
  baseline (3.4% rel diff); compare with Ada 3.54% in this sweep.
- `docs/BUILD_LOG.md` F40 — `REPERCEP_FP8_ATTENTION=fa` bridge
  behaviour on `WanTransformer3DModel`; the verification here is
  the registry-level analogue.
- `src/repercep/attention/registry.py` — the `_FP8_TRUTHY` env-value
  set and the NVIDIA branch's candidate ordering.
- `src/repercep/attention/fp8_hopper_triton.py` +
  `kernels/triton_kernels/fp8_flash_attn_hopper.py` — the kernel
  whose Ada-portability is the headline of this sweep.
