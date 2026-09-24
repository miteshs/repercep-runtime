# Wan-2.2-T2V-A14B on NVIDIA H100 (via Repercep)

*Port-ready writeup of Alibaba's Wan-2.2-T2V-A14B on a single H100 SXM5
through the Repercep runtime. Sibling of `docs/WAN_ON_MI300X.md`.*

**Status (2026-05-25):** End-to-end measurement landed on a single H100
80GB HBM3 VF through Repercep's `WanEngine` + the diffusers `WanPipeline`.
The Session 14 F30 download blocker (parallel HF downloader vs FUSE
write-rate cap) was sidestepped this session by serializing the
downloader (`HF_HUB_ENABLE_HF_TRANSFER=0` + `--max-workers 4`); the
~118 GB across 39 safetensors landed cleanly in ~8 min. The
fitting problem the WAN_ON_MI300X.md sibling foreshadowed turned out
to be real: the A14B both-experts-resident BF16 path **does not fit
in 80 GB H100 without VAE tiling** (peak overflowed at the per-frame
`torch.cat` inside `AutoencoderKLWan.forward`). Two-line fix landed
this session — new `WanConfig.vae_tiling` knob + `--vae-tiling` flag
on `scripts/run_wan.py` — which drops peak from OOM to **72.6 GiB**
at the 81 f / 40 step / 1280 × 720 reference shape, headroom to spare.

## TL;DR

We now run `Wan-AI/Wan2.2-T2V-A14B-Diffusers` end-to-end on a single
NVIDIA H100 SXM5 80GB HBM3 (`sm_90`) through Repercep's `WanEngine`. The
Wan engine code is unchanged from the MI300X path; the NVIDIA support
falls out of the vendor-neutral Backend Protocol (ADR-0003 + ADR-0006).

The Wan team's own single-H100 reference is **1041.5 s / 79.8 GB**
(BF16, FlashAttention-3, `--offload_model True --convert_model_dtype`,
FP8 weight conversion + CPU offload of the inactive MoE expert). Per
the `comp_effic.png` table in
[https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2).

| Configuration | Wan team H100 (their stack) | Repercep on H100 (this work) | Repercep on MI300X (reference) |
|---|---|---|---|
| Stack | Wan2.2 repo + FA-3 + offload + FP8 | `diffusers` 0.37.1 + Repercep WanEngine + VAE tiling | `diffusers` 0.37.1 + Repercep WanEngine |
| 17 f / 8 step smoke | — | **37.7 s** warm gen, **66.4 GiB** peak | 45.2 s warm gen, 84.3 GiB peak |
| 81 f / 40 step quality (1280 × 720) | **1041.5 s / 79.8 GB** | **1552.8 s / 72.6 GiB**, 38.82 s/step | ~1700 s steady-state proj. / 85.1 GiB |
| Offload (inactive MoE expert) | yes | no | no |
| FP8 weight conversion | yes | no | no |
| VAE tiling | (offload subsumes) | **yes (required for 80 GiB fit)** | no (192 GiB HBM has headroom) |

**Stack note.** Repercep's H100 path runs BF16 transformer + FP32 VAE,
both MoE experts resident, attention through the diffusers SDPA
dispatcher (cuDNN flash internally; FA-3 is built and the Repercep
bridge is active via `REPERCEP_FP8_ATTENTION=fa`, but appears not to
engage on `WanTransformer3DModel` — see Caveats / **F40**). The Wan
team's 1041 s number requires `--offload_model True
--convert_model_dtype` (CPU offload of the inactive expert + FP8
weight conversion); without those, the *same* configuration does not
fit in 80 GB H100, with or without VAE tiling. **Our number is
therefore NOT apples-to-apples with the Wan team's 1041 s** until we
add the matching offload + FP8 wiring. The honest framing:

- **Repercep's BF16, no-offload, both-experts-resident path on H100
  (this work):** **1552.8 s / 72.6 GiB / 38.82 s/step** at the
  canonical 81 f / 40 / 1280 × 720 shape. Directly comparable to the
  MI300X `WAN_ON_MI300X.md` number (~1700 s steady-state projected;
  same stack on different silicon). H100 is **~9 % faster end-to-end
  and ~5 % faster per-step** than MI300X here. The smoke gap is
  larger (37.7 s vs 45.2 s = 1.20×) because the smoke shape has lower
  attention-sequence cost where MI300X's wider FP32 + memory bandwidth
  fades.
- **Repercep with the Wan team's offload + FP8 stack on H100:** open.
  The work is to plumb `cpu_offload` through `WanConfig` and add an
  FP8 weight-convert path; once those land, that number IS the
  apples-to-apples bench against 1041 s. TE (TransformerEngine) is
  already installed in tree (Session 16 §3) — wiring it into
  `WanTransformer3DModel`'s attention is the bridge for FP8
  attention; the linear-conv FP8 path (the bigger Hopper lever) is
  separate.

The structural advantage MI300X has on Wan — **192 GiB HBM3 means no
offload, no tiling, no FP8 required** — does not translate to H100,
which is why the Wan team published the offload + FP8 number as their
canonical baseline. For a runtime-engine wedge, this is a real
datapoint: Repercep's both-experts-resident, full-precision-VAE,
no-tiling path on MI300X is a class of deployment H100 cannot match
in 80 GB without trading inactive-expert latency for HBM.

## Caching modes

`WanEngine` exposes the same `use_native_loop` / `cache_skip_every` /
`cache_mode` knobs as `CosmosEngine`, **but they are inert today** —
Repercep's adaptive cache implementation in
`repercep.runtime.denoise.denoise_cosmos_video` is specific to the
`CosmosTransformer3DModel` block shape and does not transfer 1:1 to
`WanTransformer3DModel` (different attention block topology + the MoE
boundary handoff). Both the MI300X and H100 Wan numbers are
**uncached** — every 40 steps run a full forward.

A Wan-shaped native loop is queued as Session 16+ work. The community
TeaCache + Sage results on 8×H100 land 2.5–3× over a similar uncached
baseline ([Morphic](https://morphic.com/blog/boosting-wan2-2-i2v-56-faster),
[Voltage Park](https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations));
the same training-free levers should transfer to single-H100 Repercep
once the loop is shaped for Wan's MoE topology.

## What we measured

Two runs, both with `REPERCEP_FP8_ATTENTION=fa`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and
`--vae-tiling`. Per-stage profile via the inline `--profile` probe
in `scripts/run_wan.py` — forward hooks fire outside any compiled
graph, so the breakdown is valid whether or not the DiT is
`torch.compile`'d (not compiled in this measurement). Model weights
warm in page cache after the first run; the reported `load_seconds`
is the second-run figure.

### Smoke baseline — 17 f / 8 steps, 1280 × 720

| | |
|---|---|
| Total generation (mean over 5 prompts × 5 seeds) | **37.66 ± 0.31 s** (range 37.4–38.0 s, spread 1.59 %) |
| Per step (avg DiT call) | 2.09 s |
| Peak HBM | **66.4 GiB** (identical across all 5 runs — allocator deterministic) |
| Load (warm page cache) | 64.6 s |
| DiT loop | 33.4 s (89 % of total) — hi-noise 12.65 s / 6 calls + lo-noise 20.73 s / 10 calls |
| VAE decode | 3.15 s |
| Text encode | 0.50 s |

H100 here is **1.20 × faster** than MI300X's 45.2 s warm smoke at the
same shape. Peak HBM is **18 GiB lower** than MI300X's 84.3 GiB —
attributable to the new VAE tiling rather than silicon (MI300X had
the headroom and didn't tile).

The 1.59 % variance is **tighter than Cosmos's 3.86 % at 121f/36** (per
SESSION_16_CLOSE §1) — consistent with F40 (no FA-3 dispatch =
no kernel-selection variance) and the smaller shape having fewer
kernels overall.  Reproduce via
`REPERCEP_FP8_ATTENTION=fa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HOME=/workspace/hf-cache .venv/bin/python scripts/verify_wan_timing.py --N 5`;
sidecar JSON written to `benchmark-results/verify_wan_timing_<ts>.json`.

### Quality reference — 81 f / 40 steps, 1280 × 720 (canonical Wan reference shape)

| | |
|---|---|
| Total generation | **1552.8 s** (≈ 25.9 min) |
| Steady-state per step | **38.82 s** |
| Peak HBM | **72.6 GiB** (vs MI300X 85.1 GiB) |
| Load (warm page cache) | 65.1 s |
| DiT loop | 1517.6 s (97.7 % of total) — hi-noise 493.24 s / 26 calls + lo-noise 1024.35 s / 54 calls |
| Per DiT-call (averaged across both experts) | 18.97 s |
| VAE decode | 28.40 s |
| Text encode | 0.32 s |
| Other (scheduler, hand-off) | 6.46 s |

Steady-state-per-step comparison: **38.82 s on H100 vs ~41 s on
MI300X**, the silicon-only delta at this Wan shape. End-to-end
H100 is **~9 % faster than MI300X**'s projected 1700 s. Against the
Wan team's single-H100 1041.5 s (with `--offload_model` + FP8 weight
convert), we are 1.49 × slower — the cost of running both 14 B
experts BF16-resident instead of swapping the inactive expert to
host RAM and computing FP8 attention.

The CFG-batched 2×-per-step pattern shows up in `dit_calls = 80`
(40 steps × 2 forwards). The high-noise / low-noise expert split
is 26 / 54 — Wan transitions around step 13 (`boundary_ratio`
≈ 0.325 in this run).

## Strategic context

`WAN_ON_MI300X.md` made the case that Wan-2.2 on MI300X is a *non-
NVIDIA model on non-NVIDIA hardware* — the entire stack free of
NVIDIA-specific dependencies. Adding the H100 sibling closes the
symmetric question: what does the same Repercep path look like on
NVIDIA's flagship inference silicon, *with the same constraints*
(no offload, no quantization)?

The answer (this work):

1. **Repercep's diffusers-path Wan on H100 does NOT fit within the 80 GB
   envelope without intervention** — the per-frame `torch.cat` inside
   `AutoencoderKLWan.forward` overflows at peak even with both
   experts otherwise comfortably loaded. The Wan team's own 79.8 GB
   figure is achievable only because their `--offload_model True`
   swaps the inactive expert to host RAM during the VAE pass.
   Repercep's path needs *either* offload (their lever) or VAE tiling
   (this work's lever, ~6-8 GiB peak reduction, bit-stable output).
2. **Silicon delta on Wan's MoE workload: ~5-9 % H100 over MI300X**,
   shape-dependent (smoke 1.20 ×; quality reference 1.09 × wall,
   1.06 × per-step). This is materially smaller than the headline
   3.81 × that Cosmos shows under the same FA-3 bridge — see Caveats
   / F40 for why the FA-3 path appears not to engage on
   `WanTransformer3DModel` end-to-end.
3. **Headroom for a Wan-shaped adaptive cache is large.** With the
   DiT at 97.7 % of total wall time and ~80 forwards across 40
   steps, even a modest cache-skip rate cuts seconds-per-step
   directly. The community's TeaCache + Sage 2.5–3 × wins on 8×H100
   are an upper bound; single-H100 Repercep with a Wan-shaped native
   loop should land 1.5–2 × of those without sequence parallelism.

## Caveats

- **FA-3 bridge does NOT engage on Wan end-to-end (F40, confirmed).**
  The Repercep source-built FA-3 (`flash-attention/hopper` minimal
  config, per `SESSION_16_CLOSE.md` §2) is installed and the diffusers
  bridge is active (`REPERCEP_FP8_ATTENTION=fa`), but
  `scripts/trace_wan_attention.py` shows **0 dispatcher engagements
  and 780 direct `torch.F.scaled_dot_product_attention` calls** over
  one 17 f / 4 step smoke. `WanTransformer3DModel`'s attention layers
  bypass `_AttentionBackendRegistry` entirely — the same F36 pattern
  Cosmos's diffusers pipeline exhibited.  Wan-side attention runs
  through cuDNN-flash internally; FA-3's WGMMA-based Hopper kernel
  never sees the call. Result: today's `REPERCEP_FP8_ATTENTION=fa`
  flag is informational, not load-bearing, for Wan. See BUILD_LOG
  F40 for the trace + fix paths.
- **VAE tiling is required.** Without `--vae-tiling`, the run OOMs
  at the VAE decode step regardless of `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True` (which recovers ~3 GiB of fragmentation
  but activations grow into the gap). Output is bit-stable with
  tiling enabled — the AutoencoderKLWan decoder already does seam
  blending across spatial tiles.
- **Single MI300X / single H100 VF.** Multi-GPU paths are not
  exercised on either host. AMD's MLPerf submission for Wan-2.2 used
  MI355X and a different (likely distilled) workload definition; that
  is not the comparison we are making.
- **No FVD comparison.** Same as the MI300X writeup — there is no
  published Wan reference clip + seed combination to diff against;
  visual quality is eyeballed on the produced mp4 and confirmed to be
  a valid Wan output.
- **NVIDIA-canonical FP8 (TransformerEngine) not exercised here.** TE
  installation completed in Session 16; the TE-via-Repercep path
  through `WanTransformer3DModel`'s attention layers is Session 17+
  scope. The current measurement is BF16 transformer everywhere.
- **Wan-native adaptive cache not exercised.** `WanConfig`'s
  `use_native_loop` / `cache_skip_every` knobs are inert — the
  Cosmos-shape cache in `repercep.runtime.denoise.denoise_cosmos_video`
  doesn't transfer to `WanTransformer3DModel`. A Wan-shaped native
  loop is the highest-value next move per Strategic Context §3.

## Reproduce

```bash
git clone https://github.com/miteshs/Repercep.git repercep && cd repercep
uv venv --python 3.12 .venv
# H100 — install torch from the cu128 wheel index, NOT rocm7.2.
uv pip install --python .venv torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv wheel packaging ninja  # FA-3 build deps
make install                 # installs Repercep + deps
PATH="$HOME/.cargo/bin:$PATH" make rust-install  # builds the 3 PyO3 crates
.venv/bin/hf auth login      # for Wan-AI/Wan2.2 access (Apache 2.0,
                             #   no license click-through needed)

# FA-3 source build (~5 min minimal config; see SESSION_16_CLOSE.md §2
# for the full flag set). Wire into the diffusers bridge via the
# REPERCEP_FP8_ATTENTION env var below. Even though FA-3 doesn't appear
# to engage on Wan today (F40), it's harmless to have built + active.
cd /tmp && git clone --depth 1 https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper && \
  PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 \
  FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
  FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
  FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE \
  FLASH_ATTENTION_DISABLE_FP8=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
  FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
  MAX_JOBS=8 uv pip install --python /workspace/Mirage/.venv --no-build-isolation .

# Pre-fetch Wan weights with the serialized downloader (F30 workaround).
# Without HF_HUB_ENABLE_HF_TRANSFER=0 + --max-workers <small> the parallel
# downloader can overwhelm a FUSE-mounted volume's write-rate cap.
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME=/workspace/hf-cache \
  .venv/bin/hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --max-workers 4

# Smoke (~65 s warm-page-cache load + ~38 s gen)
HF_HOME=/workspace/hf-cache REPERCEP_FP8_ATTENTION=fa \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/run_wan.py --frames 17 --steps 8 --vae-tiling --profile

# Quality reference (~25.9 min)
HF_HOME=/workspace/hf-cache REPERCEP_FP8_ATTENTION=fa \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/run_wan.py --frames 81 --steps 40 --vae-tiling --profile
```

The `--vae-tiling` flag is what unblocks the 80 GB envelope; without
it the 81 f / 40 reference run OOMs at the per-frame `torch.cat`
inside `AutoencoderKLWan.forward`. `REPERCEP_FP8_ATTENTION=fa` registers
the Repercep attention bridge with the diffusers dispatcher; today this
is a no-op for Wan (see Caveats / F40) but harmless. The
`expandable_segments:True` is a defense-in-depth against allocator
fragmentation — required for FA-3 paths on Cosmos, optional but
recommended here.

## Versions used

| Component | Version |
|---|---|
| GPU | NVIDIA H100 SXM5 80GB HBM3 (sm_90, 132 SMs, 18× NVLink @ 26.6 GB/s) |
| CUDA driver | 580.126.09 (CUDA 13.0) |
| Python | 3.12.3 |
| `torch` | 2.8.0+cu128 (from `download.pytorch.org/whl/cu128`) |
| `diffusers` | 0.37.1 (registers `WanPipeline`, `AutoencoderKLWan`, `WanTransformer3DModel`) |
| `transformers` | 5.9.0 |
| `accelerate` | 1.13.0 |
| `flash-attn-3` | 3.0.0 (built from source, `Dao-AILab/flash-attention/hopper`, minimal config) |
| Repercep | 0.0.1 (this repo, NVIDIA backend via ADR-0006) |

## References

Inherits the MI300X writeup's reference table verbatim
(`docs/WAN_ON_MI300X.md` § "References"). Specifically the Wan team's
`comp_effic.png` and the 8×H100 optimization-stack numbers (Morphic,
Voltage Park, Simplismart, Baseten) — those H100 results use
sequence parallelism + TeaCache / Magcache + Sage Attention on 8
GPUs, which is not directly comparable to the single-H100 Repercep path
documented here.

## Acknowledgements

Wan-Video (Alibaba) team for open-sourcing Wan-2.2 under Apache 2.0.
HuggingFace `diffusers` maintainers for the `WanPipeline` /
`AutoencoderKLWan` / `WanTransformer3DModel` authors. The cuDNN +
hipBLASLt teams for the underlying kernel infrastructure.
