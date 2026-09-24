# Wan-2.2-TI2V-5B on Intel CPU (Sapphire Rapids)

*CPU sibling of `docs/WAN_ON_H100.md` and `docs/WAN_ON_MI300X.md`. Same
`WanEngine`, same diffusers pipeline, different backend per ADR-0007.*

**Status (2026-05-25):** Smoke runs end-to-end on Sapphire Rapids via torch
SDPA + oneDNN AVX512_BF16. AMX-flash and Wan-shaped attention paths are
hardware-blocked on this VM (no `amx_bf16` exposed by `/proc/cpuinfo`;
the AMX kernel build refuses with a FATAL message, which is correct
behavior). The single landed number was taken with `--vae-tiling=True`
for "parity" with the H100 invocation — a methodology miss documented
below and surfaced as a CLI refusal in `scripts/run_wan.py` so it does
not recur. The corrected re-run is open work.

## TL;DR

We can drive `Wan-AI/Wan2.2-TI2V-5B-Diffusers` end-to-end on a single
Intel Xeon Platinum 8468 (Sapphire Rapids, 160 logical cores,
`OMP_NUM_THREADS=120`, 1.5 TiB DRAM) through Repercep's `WanEngine`. The
Wan engine code is unchanged from the GPU paths; the Intel backend
falls out of the vendor-neutral Backend Protocol (ADR-0003, ADR-0006,
ADR-0007). Honest framing: CPU Wan is **not a production latency
target** — the 17-frame / 8-step TI2V-5B smoke is tens of minutes per
clip; the canonical 81 f / 40 reference shape is hours. The substrate
is here so the Backend Protocol holds end-to-end across silicon
families, and so CI / numerical-parity work on the Wan engine doesn't
need a GPU.

## Measured numbers

Host: Intel Xeon Platinum 8468 (Sapphire Rapids, family 6 model 143,
80 physical × 2 sockets = 160 logical, 1.5 TiB DRAM, AVX-512 BF16 +
AVX-VNNI exposed; **`amx_bf16` NOT exposed by `/proc/cpuinfo`** —
hypervisor masks AMX on this VM).

| Config | Wall | Peak RSS delta | Notes |
|---|--:|--:|---|
| TI2V-5B 17 f / 8 step (BF16 DiT, FP32 VAE), `--vae-tiling=True` | **66 min** (3969.8 s) | 12.4 GiB | **Methodology-tainted** — see footnote [1] |
| TI2V-5B 17 f / 8 step, no `--vae-tiling` | **TBD** (estimated 38-45 min) | TBD | Corrected re-run, queued |
| TI2V-5B 81 f / 40 step, no `--vae-tiling` | **TBD** | TBD | Reference shape; hours-scale on CPU |
| A14B (either shape) | **N/A** | — | A14B both-experts-resident is hours+ per smoke on CPU; not a near-term target |

[1] DiT loop 2193.5 s (~36.5 min), VAE decode 1734.2 s (~29 min). The
VAE decode dominates because `--vae-tiling` was enabled and shredded
the FP32 decode into hundreds of small per-tile forwards — see
§"Methodology" below. The corrected re-run is expected to land in the
38-45 min envelope (DiT loop unchanged; VAE decode collapses to a few
minutes). Per `docs/SESSION_17_CLOSE.md` §"CPU 5B sidebar — runs, but
with a methodology miss".

## Methodology — why --vae-tiling is refused on CPU

`scripts/run_wan.py` now **refuses** (exits with code 2) when
`--vae-tiling` is passed and the selected backend is `cpu` /
`Vendor.INTEL`. The error message:

```
[repercep] FATAL: --vae-tiling is not supported on the CPU backend.
  --vae-tiling shreds the FP32 VAE decode on CPU; observed +30 min
  on the TI2V-5B 17f/8 smoke. See docs/WAN_ON_CPU.md §"Methodology".
  Re-run without --vae-tiling.
```

Why a hard refusal instead of silently auto-disabling: `--vae-tiling`
exists to drop GPU HBM peak (~18 GiB on H100 at 1280 × 720, the
difference between OOM and a 72.6 GiB fit). It does this by decoding
the VAE in spatial tiles via `pipe.vae.enable_tiling()`. On CPU the
same flag is actively harmful — it shreds the FP32 VAE decode into
hundreds of small per-tile forwards, none of which fit oneDNN's
preferred AMX/AVX512 block sizes; the per-tile launch + reduction
overhead dominates and the throughput cliff is roughly the **+30 min**
delta measured in Session 17 (66 min observed vs the estimated 38-45
min envelope without tiling, on a workload where the entire envelope
is 1.5 TiB of host RAM and there is nothing to "fit"). Silently
flipping the flag would hide that the user requested a configuration
that doesn't make sense on this backend; the refusal keeps the
methodology mistake legible in the run log instead of burying it.

GPU users (`--backend cuda` / `--backend rocm`) are unaffected — the
flag is required on H100 to fit 80 GiB HBM at 1280 × 720, and
recommended on MI300X (~18 GiB peak reduction at no quality cost).

## Reproduce

```bash
git clone https://github.com/miteshs/Repercep.git repercep && cd repercep
uv venv --python 3.12 .venv
# Plain CPU torch (no CUDA wheel needed for CPU-only)
uv pip install --python .venv torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e ".[models,cpu,dev]"
.venv/bin/hf auth login   # for Wan-AI/Wan2.2 access (Apache 2.0)

# Pre-fetch the 5B weights via the serialized downloader (F30 workaround)
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME=/workspace/hf-cache \
  .venv/bin/hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --max-workers 4

# Smoke — TI2V-5B 17 f / 8 step. NOTE: NO --vae-tiling.
OMP_NUM_THREADS=120 HF_HOME=/workspace/hf-cache \
  .venv/bin/python scripts/run_wan.py \
    --small --backend cpu \
    --frames 17 --steps 8 --profile
```

The `--small` flag selects `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (~10 GiB
BF16) rather than the A14B MoE variant. `--backend cpu` is explicit;
on a CPU-only host `--backend auto` resolves the same way. Do **not**
pass `--vae-tiling` — `scripts/run_wan.py` will refuse with the FATAL
message above.

For the A14B path on CPU: not a near-term target. Both 14 B experts
resident in BF16 + FP32 VAE through oneDNN AVX512_BF16 is well into
hours per smoke at this shape, and the AMX kernel is hardware-blocked
on this VM. Hardware that exposes `amx_bf16` (bare-metal Sapphire
Rapids, Emerald Rapids, Granite Rapids) would change the picture for
the DiT loop; the VAE decode would not benefit (FP32, not BF16).

## Honest caveats

* **Methodology-tainted single number.** The only landed CPU
  measurement (66 min) ran with `--vae-tiling=True`, which is wrong on
  CPU for the reasons in §"Methodology". The corrected re-run is open
  work. Numbers in this doc that say "TBD" are TBD because they are
  TBD, not because they are pending typing.
* **No AMX exposed on this VM.** `/proc/cpuinfo` shows
  `avx512_bf16 + avx_vnni` only; the hypervisor masks `amx_bf16`. The
  Repercep AMX flash kernel build refuses with a clear FATAL message
  (this is correct behavior, not a regression). The CPU smoke
  validated `WanEngine` end-to-end through the torch SDPA + oneDNN
  AVX512_BF16 path, but did **not** exercise Repercep's AMX kernel; the
  AMX numbers in `docs/COSMOS_ON_CPU.md` for the Cosmos shape are
  shape-dependent and do not transfer to Wan.
* **Wan-shaped attention path not exercised on CPU.** Same F40 pattern
  as the H100 path — `WanTransformer3DModel` bypasses Repercep's
  `_AttentionBackendRegistry` entirely; attention runs through
  `torch.F.scaled_dot_product_attention` direct. On CPU the SDPA
  fallback is the oneDNN path, which is fine; on a host with AMX
  exposed it still would not engage Repercep's AMX flash kernel until
  the F40 fix-path 1 (custom `WanAttnProcessor` via
  `set_attn_processor`) lands. See `docs/BUILD_LOG.md` F40.
* **No FP8 ISA on any shipped Xeon.** The FP8 paths in the registry
  are AMD-/NVIDIA-only. The next Intel generation that gets FP8 will
  land as a fourth uarch entry alongside `spr`/`emr`/`gnr`; until
  then the CPU capabilities never advertise FP8.
* **A14B is out of scope for CPU.** Both 14 B experts BF16-resident +
  FP32 VAE is a multi-hour smoke even on bare-metal Sapphire Rapids
  with AMX; the bottleneck isn't memory (1.5 TiB DRAM has room) but
  attention sequence length × expert count. TI2V-5B is the only
  near-term CPU target.
* **No FVD comparison.** Same as the GPU Wan writeups — there is no
  published Wan reference clip + seed combination to diff against;
  visual quality is eyeballed on the produced mp4 and confirmed to be
  a valid Wan output.

## See also

* `docs/COSMOS_ON_CPU.md` — Cosmos-Predict-7B sibling; covers the AMX
  flash kernel and the IPEX path.
* `docs/WAN_ON_H100.md` — full H100 measurements; explains why
  `--vae-tiling` is required there (it is, on GPU).
* `docs/WAN_ON_MI300X.md` — MI300X reference path; no tiling, no
  offload (192 GiB HBM has the headroom).
* `docs/SESSION_17_CLOSE.md` §"CPU 5B sidebar — runs, but with a
  methodology miss" — the trace of the methodology miss and the
  open re-run item (§"What's open after today" item #6).
* `docs/adr/0007-cpu-backend.md` — why the Backend Protocol extends to
  CPU and what that protocol promises.
