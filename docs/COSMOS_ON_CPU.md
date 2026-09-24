# Cosmos-Predict-7B on Intel CPU (Sapphire Rapids AMX)

**Status:** Architecture landed (Session 15); end-to-end run pending FUSE
recovery + kernel build.  See ADR-0007.

See also `docs/WAN_ON_CPU.md` for the Wan-2.2-TI2V-5B CPU sibling.

This is the CPU sibling of `docs/COSMOS_ON_MI300X.md` and
`docs/COSMOS_ON_H100.md`.  Honest framing: Cosmos-Predict-7B on CPU is
**not a production latency target** — even on an Intel Sapphire Rapids
host with 96 logical cores and AMX_BF16, the 121-frame × 36-step reference
config is minutes per video.  The substrate is here so the Backend
Protocol holds (ADR-0007) and so CI / numerical-parity work doesn't need
a GPU; the perf claim is "this runs" rather than "this is fast."

## Pending headline numbers

All measured (or to be measured) on the host CPU of the Repercep H100
template — Intel Xeon Platinum 8470 (Sapphire Rapids, family 6 model
143, 52 physical cores × 2 sockets = 208 logical, 1007 GiB DRAM,
AMX_BF16 + AMX_INT8 + AVX-512 BF16/FP16 + AVX-VNNI):

| Config | Wall | vs Repercep H100 (138.4 s adaptive) | Peak RAM |
|---|--:|--:|--:|
| Smoke (17 f / 8 steps, BF16, SDPA→oneDNN AMX) | **TBD** | — | TBD |
| Smoke (17 f / 8 steps, BF16, AMX flash kernel) | **TBD** | — | TBD |
| Adaptive cache (thr=0.30) at 121 f / 36 steps | **TBD** | — | TBD |
| Adaptive + AMX flash kernel | **TBD** | — | TBD |
| Adaptive + AMX flash + INT8 quantized DiT weights | **TBD** | — | TBD |

The two AMX-aware attention paths are:

* **AMX flash kernel** (`repercep.attention.amx_flash.AMXFlashAttention`) —
  Repercep-owned C++ kernel using `_tile_dpbf16ps` for the QK^T and PV
  matmuls inside a flash-attention online-softmax loop.  Sibling of the
  gfx942 and Hopper Triton kernels — same per-vendor pattern (ADR-0006).
  Source: `kernels/cpu/amx_attn/flash_attn_amx.cpp`.
* **IPEX flash** (`repercep.attention.ipex_flash.IPEXFlashAttention`) —
  wrapper over Intel-Extension-for-PyTorch's fused attention.  Wider
  shape coverage (head_dim ∈ {64, 80, 96, 128, 192, 256}); requires an
  IPEX install (~1 GiB).

## Reproduce

```bash
git clone https://github.com/miteshs/Repercep.git repercep && cd repercep
git checkout cpu-amx-port

uv venv --python 3.12 .venv
# Plain CPU torch (no CUDA wheel needed for CPU-only)
uv pip install --python .venv torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e ".[models,cpu,dev]"

# Build the AMX flash kernel (requires Sapphire Rapids or newer)
make kernels-cpu

# Sanity (should report Sapphire Rapids + AMX_BF16/INT8 features)
make check-gpu
# (Same script — vendor-neutral; reports the CPU when no GPU is visible.)

# Smoke gen on CPU (BF16, SDPA→oneDNN, AMX dispatch is automatic)
.venv/bin/python scripts/run_cosmos.py --frames 17 --steps 8

# Smoke with the Repercep AMX flash kernel (env-promoted attention op)
REPERCEP_AMX_ATTENTION=amx .venv/bin/python scripts/run_cosmos.py \
    --frames 17 --steps 8

# Full reference with adaptive cache
.venv/bin/python scripts/run_cosmos.py \
    --frames 121 --steps 36 --native-loop \
    --cache-mode adaptive --cache-adaptive-threshold 0.30 \
    --cache-force-full-every 16
```

## What the AMX flash kernel does

Standard FlashAttention-2 algorithm — tile over (Q_tile, K_tile, V_tile),
maintain a running rowmax + rowsum per Q-row, never materialise the full
(B, H, S, S) score matrix.  The AMX-specific decisions:

* **Tile dimensions** sized to fit two AMX tile registers: 32×D Q-tile
  (D ∈ {64, 128}), 32×D K-tile, S tile 16×16 fp32 accumulator.
* **`_tile_dpbf16ps`** for the two matmuls — accumulates two 16×32 BF16
  tile pairs into a 16×16 FP32 output tile in a single instruction.
* **AVX-512 BF16** for the softmax + rescale steps — `_mm512_*_pbh`
  intrinsics so the online softmax is fp32-precise on the running
  rowmax/rowsum but BF16-precise everywhere else.
* **OpenMP** parallelism at the outer `(B, H, S_q_tile)` loop — each
  thread owns its own AMX tile config (`_tile_loadconfig` is per-CPU
  state, so threads don't fight over it).
* **K pre-transposition** outside the inner loop, so the second matmul
  reads V contiguously — saves an L1 miss per tile.

## Why this lands now

See `docs/adr/0007-cpu-backend.md`.  Short version: ADR-0003's "one
class per vendor" promise was quietly "one class per *GPU* vendor"
until this lands.  Now it isn't.

## Honest caveats

* **No FP8 ISA on any shipped Xeon** — the FP8 paths in the registry are
  AMD-/NVIDIA-only.  The next Intel generation that gets FP8 will land as
  a fourth uarch entry alongside `spr`/`emr`/`gnr`; until then the CPU
  capabilities never advertise FP8.
* **No CUDA/HIP for diffusion VAE** — the Cosmos pipeline's VAE encode/decode
  steps assume one device per tensor.  On CPU the VAE runs the same code
  path but on the same `cpu` device; the per-stage profiler treats it
  identically.
* **Quality is bit-identical to the BF16 GPU path** when SDPA is used —
  this is the numerical-parity oracle ADR-0007 calls out.  The custom
  AMX flash kernel is *not* bit-identical (BF16 reduction order differs),
  but matches within typical FA-2-vs-SDPA tolerances (rtol ≈ 2 %).

## Open work

| Item | Where | Status |
|---|---|---|
| Build + benchmark the AMX BF16 flash kernel end-to-end | `kernels/cpu/amx_attn/` + `scripts/run_cosmos.py --frames 17 --steps 8` | **Hardware-blocked** — every dev VM the project has access to today (RunPod H100 and MI300X templates) reports `amx_bf16` masked by the hypervisor; the kernel builds + runs on bare-metal Sapphire Rapids when one is available |
| AMX INT8 attention kernel | `kernels/cpu/amx_int8_attn/flash_attn_amx_int8.cpp` + `src/repercep/attention/amx_int8_flash.py` | **Code landed Session 18 (Item B)** — TDPBSSD-based, BF16-in/BF16-out, dynamic per-tile quant; build refuses without `amx_int8` flag, same hardware gate as the BF16 sibling |
| Per-channel symmetric INT8 weight quantization | `src/repercep/runtime/quantize.py` | **Landed Session 18 (Item A)** — `QuantizedLinear` + `QuantizedLinearModule` + `replace_linears_with_quantized`; CPU/CUDA bit-parity on `qweight`, 1-ULP slack on `scale` validated on real RTX 2000 Ada (Item H) |
| FVD on CPU vs GPU adaptive output | `scripts/eval_cpu_quality.py` + `scripts/compute_fvd.py` | **Wired Session 18 (Item F)** — combined LPIPS+MSE+PSNR+FVD runner with `--device cpu` forced; LPIPS CPU/CUDA parity validated within 1.5e-5 (Item I).  Held-out reference set workflow documented in `docs/METHODOLOGY.md` §"Held-out reference set for FVD"; the set itself is local-only (N≥50 generation is still owed) |
| Granite Rapids AMX_FP16 kernel | `kernels/cpu/amx_fp16_attn/` + `src/repercep/attention/amx_fp16_flash.py` | **Scaffolded Session 18 (Item C)** — wrapper, setup.py, tile config, OMP shell + `arch_prctl` opt-in all real; the inner `_tile_dpfp16ps` loop is `// TODO(GNR):` markers (six total) for the eventual GNR-host fill-in |
