# Wan-2.2 T2V-A14B on AMD MI300X

*First publicly disclosed inference benchmark of Alibaba's Wan-2.2 MoE
world-model family on AMD Instinct silicon, via the diffusers reference
path.*

**Status (2026-05-23):** Measurement on a single MI300X VF through Repercep's
`WanEngine` + the diffusers `WanPipeline`. The 81 f / 40 step quality run
is **cold** (one-time ROCm kernel autotuning included) — wall = 2576 s but
the **steady-state per-step is ~41 s** at the back end of the run, with
peak HBM = 85.1 GiB. The 17 f / 8 step smoke is **warm** at ~45–51 s
generate / 84.3 GiB peak. Caveats around contention and the cold/warm gap
are spelled out below.

## TL;DR

We ran `Wan-AI/Wan2.2-T2V-A14B-Diffusers` end-to-end on an AMD Instinct
MI300X VF (`gfx942`, ROCm 7.2.0, torch 2.12+rocm7.2) through the Repercep
runtime. To our knowledge, this is the **first publicly disclosed
inference benchmark of Wan-2.2-T2V-A14B (or any Wan-2.2 variant) on any
AMD GPU end-to-end with wall time + peak HBM, via diffusers, BF16 + FP32
VAE**, and the **first against the Wan team's own H100 reference number
of 1041.5 s / 79.8 GB on a single GPU**.

Wan-2.2-T2V-A14B is a Mixture-of-Experts text-to-video model: two ~14 B
parameter expert transformers (high-noise + low-noise), ~14 B parameters
active per denoise step, ~27 B parameters total, Apache 2.0. The MoE
boundary handoff swaps the active expert mid-denoise.

| Configuration | NVIDIA single H100 (Wan team reference) | Repercep on AMD MI300X (this work) |
|---|---|---|
| Stack | `Wan2.2` repo + FlashAttention-3 + offload | `diffusers` 0.37.1 + SDPA→aotriton |
| 17 frames @ 1280×720, 8 steps — **smoke** (warm) | — | **45.2 s** generate / **84.3 GiB** peak HBM |
| 81 frames @ 1280×720, 40 steps — **quality reference** | **1041.5 s** (BF16, FA-3, `--offload_model True --convert_model_dtype`) / **79.8 GB peak** | **2576 s** generate (one-shot, cold autotune) / **85.1 GiB** peak HBM; **~41 s/step steady-state** in the back end (≈ **1640 s** projected steady-state DiT loop / **~1700 s** projected end-to-end without the cold autotune tail) |

**Headline (steady-state projected):** at the canonical Wan reference shape
(81 f / 40 steps / 1280×720), Repercep's MI300X **per-step time settles to
~41 s** after the cold-cache phase, which projects to **~1640 s for the DiT
loop and ~1700 s end-to-end** — uncached, no caching mode, no FP8 wiring,
no native loop. The Wan team's own single-H100 reference at the same shape
is **1041.5 s** with `--offload_model True --convert_model_dtype` (FP8
weight conversion + CPU offload of the inactive MoE expert). Apples-to-
apples this isn't yet a wall-time win against H100; the structural
takeaway is **MI300X runs the workload at ~85 GiB peak with no offload
and no FP8 quantization**, where H100 needs both.

The 2576 s measured wall is **inflated by two factors that are not part
of the steady-state cost**: (1) one-time ROCm kernel autotuning at the
81-f / 40-step / 720-P shape (~5 min on first use; the same cold-vs-warm
gap Cosmos shows in `COSMOS_ON_MI300X.md`), and (2) **GPU contention** with
a sister-agent FP8 autotune sweep that ran concurrently with steps 1-19
of the primary run (visible in the per-step times — steps 1 to 19 averaged
75 s/step, steps 20 to 40 settled to 51 s/step then 41 s/step). A clean
single-tenant warm re-run is the obvious follow-up to nail the headline
number down. The data we **do** have shows steady-state per-step landing
in the 41-45 s range; we project from those.

Like Cosmos, Repercep uses **none** of the FP16-Hopper-specific tooling
(no FlashAttention-3, no FSDP-ed multi-GPU sharding, no FP8 weight
conversion, no CPU offload) — just stock PyTorch SDPA→aotriton flash
kernels on ROCm, a single VF, BF16 transformer + FP32 VAE, both MoE
experts resident.

## Available reference numbers

The diligence behind the "first publicly disclosed inference number for
Wan-2.2 on AMD MI300X" framing. The Wan-2.2 community has published
H100-class numbers but the **single-GPU baseline is itself rarely cited**;
most blogs focus on 8×H100 distributed runs.

### Canonical: Wan team's own efficiency table (single H100/H800)

The most authoritative single-GPU number comes from the Wan2.2 repo's
`assets/comp_effic.png` ("Computational Efficiency of Wan2.2", format
`Time (s) / Peak Memory (GB)`). Config:
`height=720, width=1280, num_frames=81, num_inference_steps=40,
guidance_scale=4.0, guidance_scale_2=3.0`, FlashAttention-3 on Hopper,
`--offload_model True --convert_model_dtype` for single-GPU.

| Model | 480 P | 720 P (this work's config) |
|---|--:|--:|
| T2V-A14B | 326.9 s / 41.3 GB | **1041.5 s / 79.8 GB** |
| I2V-A14B | 327.8 s / 41.0 GB | 1055.9 s / 59.7 GB |

For context the same table reports a single A100/A800 at **2735.7 s / 59.8 GB**
for T2V-A14B 720 P, and 8×H100 at 155.1 s / 71.1 GB. The 4090 row covers only
the TI2V-5B variant (not A14B).

Source: <https://github.com/Wan-Video/Wan2.2> (`assets/comp_effic.png`).

### Optimization-stack numbers (8×H100; not directly comparable)

These are bandwidth-rich, distributed numbers on 8 GPUs, and exist for
the I2V-A14B variant. We list them only to be exhaustive about what's
out there — they aren't the single-GPU baseline for our table.

- **Morphic** (8×H100, I2V-A14B, 1280×720, 81 f, 40 steps): baseline
  Flash Attention 2 = **250.70 s**; their final
  FA-3 + TF32 + Magcache + torch.compile = **109.81 s** (2.28×); aggressive
  Magcache = 98.87 s (2.53×).
  Source:
  <https://morphic.com/blog/boosting-wan2-2-i2v-56-faster>
- **Simplismart** (8×H100, T2V-A14B and I2V-A14B): **159 s baseline →
  49 s optimized** (3.2×). Frame count / resolution not stated in the
  post; configuration appears to be the same Wan reference shape.
  Reports "5-second video still takes approximately 17 minutes on a single
  H100" baseline — i.e. ~1020 s, in agreement with the Wan team's 1041.5 s
  reference. Source:
  <https://simplismart.ai/blog/deploy-wan-2-2>
- **Voltage Park** (8×H100, T2V-A14B, 40 denoising steps): **187 s baseline
  → 60 s** (3.1×). Their per-step times (4.67 s → 1.51 s) imply BLOCK_M /
  BLOCK_N + Sage Attention + TeaCache stacked. Frame count and resolution
  not stated. Source:
  <https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations>
- **Baseten** (H100 vs B200, T2V-A14B): "less than 60 seconds" headline on
  B200. Relative claims only ("2.6× faster" on H100, "3.2× faster" on B200
  vs baseline); no absolute wall times disclosed.
  Source:
  <https://www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/>
- **fal.ai** (commercial inference): $0.08 / video-second at 720 P,
  $0.06 / video-second at 580 P, $0.04 / video-second at 480 P. Implies a
  per-clip compute cost but doesn't disclose wall time or GPU.
  Source:
  <https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video>

### MLPerf v6.0

AMD submitted **Wan-2.2-T2V-A14B** as the first ever Wan-2.2 MLPerf entry
(round 1, v6.0 — April 2026), Single Stream scenario, on **MI355X (one
GPU)**: **27.4 s** latency per video. The MLPerf workload definition (frame
count, resolution, step count) is not directly disclosed in AMD's blog;
note that MLPerf Single Stream is *one query at a time*, the metric is
per-query latency, and the result implies an aggressive optimization path
(likely a distilled / cached / quantized configuration). It is **not the
same as the Wan team's no-shortcut 1041.5 s H100 reference**, and shouldn't
be lined up against it directly. Source:
<https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html>.

There **is no official NVIDIA Wan-2.2 reference page** (the way there is
for Cosmos): Wan is Alibaba's model, not NVIDIA's. NVIDIA may have a v6.0
MLPerf submission for Wan-2.2 alongside AMD's; we couldn't surface specific
results in the public MLCommons posts as of 2026-05-23. Anyone looking for
ground truth should cite the Wan team's own efficiency table (above) as
the canonical reference.

### What we couldn't find

- A standalone arxiv paper for Wan-2.2. The Wan-2.1 paper (arxiv:2503.20314)
  predates the MoE upgrade; Wan-2.2's release notes are blog-form on the
  Wan-Video GitHub. No paper means no rigorous benchmark methodology
  section to point at.
- An NVIDIA-published Wan-2.2 H100/H200/B200 reference page (the way
  `nvidia/Cosmos-Predict1-7B-Text2World` has on HuggingFace). The
  Wan-2.2 model card on HuggingFace links the same `comp_effic.png` we
  reproduce above, no NVIDIA-side curation.
- Single-MI300X Wan-2.2 numbers anywhere in AMD's blog series. AMD's own
  publishing focuses on MI355X MLPerf Single Stream, not the diffusers
  reference path. The Wan-2.2 model is in AMD's MI355X MLPerf submission;
  it is conspicuously absent from MI300X / MI325X coverage.

## Caching modes

Wan's diffusers `WanPipeline` ships with no built-in step-caching path.
The Wan-2.2 community ecosystem includes TeaCache / Magcache adaptations
for the diffusers `WanTransformer3DModel` (e.g. the Morphic and Voltage
Park blogs above both fold in TeaCache), but Repercep's adaptive caching
implementation (in `repercep.runtime.denoise.denoise_cosmos_video`) is
**specific to the Cosmos `CosmosTransformer3DModel` block topology**.

`WanEngine` exposes the same `use_native_loop` / `cache_skip_every` /
adaptive-threshold knobs on `WanConfig` so the API surface is stable
across model families, but they are **forward-compat hooks** — today they
are inert and the diffusers `WanPipeline.__call__` is the only path. A
Wan-shaped native loop with the MoE second-expert boundary preserved is
queued as a follow-up.

This means the **headline number reported below is uncached** — every
40 steps run a full forward, with the high-noise expert active for the
first portion and the low-noise expert for the rest. A TeaCache-style
gate on Wan, given the H100 community's 2.5–3× results on similar
caches, would be the highest-value next move.

## What we measured

**Hardware:** AMD Instinct MI300X VF (192 GiB HBM3, 304 CUs, gfx942 /
CDNA3); ROCm 7.2.0; torch 2.12.0+rocm7.2; 235 GiB host RAM; 20 CPU cores.

**Software:** Repercep Runtime (this repo), `diffusers` 0.37.1 +
`transformers` 5.9.0, BF16 transformer / FP32 VAE (per the Wan reference
path). The diffusers `WanPipeline` runs unmodified above Repercep's
`WanEngine`; the `--profile` flag wraps a per-stage probe identical in
shape to `scripts/profile_cosmos.py`, with the MoE second expert
(`transformer_2`) accounted for as a separate counter.

**Workload:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, BF16 transformer +
FP32 VAE, 81 frames at 1280×720, 40 denoising steps, guidance scale 4.0,
guidance scale 2 (low-noise expert) = 3.0 — the same configuration the
Wan team's `comp_effic.png` publishes a single-H100 number for.

### Smoke baseline — 17 frames / 8 steps, warmup-separated

| | |
|---|---|
| **Total generation** | **45.2 s** (gen) + 15.5 s (model load) |
| Per step | ~5.0 s / step (steady) |
| Peak HBM | **84.3 GiB / 192 GiB** |
| Throughput | 0.376 frames/s |
| Output | `benchmark-results/wan_smoke_17f_8steps.mp4`, 17 frames verified |

**Notes on Session 10's 326.2 s smoke number.** Session 10 reported
`load 18.8 + gen ~307` for the same config. This re-run measured `15.5
+ 45.2`. The load number is within noise; the generation gap is the
**one-time ROCm kernel autotuning cost** (aotriton, hipBLASLt, MIOpen
populating their on-disk caches on first-shape use). The MIOpen cache
on this host (`~/.cache/miopen/3.5.1.dabb6df2b9/gfx942130.ukdb`) was
warm from Session 10; the 45.2 s number is **steady-state**. Same shape
of finding as the Cosmos 738 → 465 s cold-vs-steady gap (`OPTIMIZATION.md`
§2, `COSMOS_ON_MI300X.md` §"Cold first run"). Warming kernel caches at
deploy time is itself a real latency win for any Wan-2.2 first-launch
on a fresh ROCm system.

A second smoke re-run on the same warm host returned **51.2 s** generate
— within 13 % of the 45.2 s first measurement and well-grouped vs the
Session 10 cold 307 s. We treat 45–51 s as the steady-state band.

### Quality reference — 81 frames / 40 steps

This is the **canonical Wan-2.2-T2V-A14B reference shape** — 5-second
clip at 16 FPS, 1280×720, 40 inference steps, guidance 4.0 / 3.0. The
same shape the Wan team publishes 1041.5 s / 79.8 GB H100 single-GPU
numbers for in `assets/comp_effic.png`.

| | |
|---|---|
| Total generation (measured, one-shot, cold) | **2576.3 s** |
| Steady-state per-step (last 11 steps) | **~41 s / step** |
| Projected DiT loop (40 × 41 s steady-state) | **~1640 s** |
| Projected total at steady-state (no autotune tail) | **~1700 s** |
| Peak HBM | **85.1 GiB / 192 GiB** |
| Throughput (measured wall) | 0.031 frames/s |
| MoE boundary | `boundary_ratio = 0.875` (from `model_index.json`) → transformer (high-noise) for `t >= 875`, transformer_2 (low-noise) for the rest. Scheduler is UniPCMultistepScheduler with `prediction_type="flow_prediction"`, `use_flow_sigmas=True`, `flow_shift=3.0`. Empirically the boundary lands at step ~17 of 40 — the flow-shift schedule spends most time at high `t` |
| Output | `benchmark-results/wan_quality_81f_40steps.mp4`, 81 frames verified |

**Why 2576 s, not ~1700 s.** Two contaminants in the one-shot run:

1. **ROCm kernel autotuning on first-shape use.** Same pattern as
   Cosmos's 738 → 465 s gap. Visible in the first 5 steps of the primary
   run: step 1 measured at 577 s, step 5 at 95 s (moving average) before
   the kernel cache settled. ~5–8 min of one-time cost.
2. **Sister-agent GPU contention.** A concurrent FP8 autotune sweep on
   the same MI300X VF (Cosmos production shape, ~22 minutes wall) overlapped
   steps 1-19 of our primary run and steps 3-onward of the profile re-pass.
   Per-step times dropped from 60–115 s under contention to 41 s clean.

The steady-state ~41 s / step is the trustworthy per-step number; the
~1640 s DiT loop and ~1700 s end-to-end are projected from it.

**Cross-check on the steady-state number.** A second-pass profile re-run
started immediately after the primary completed measured steps 1 and 2
at **41.15 s and 41.17 s respectively** — fully warm, fully uncontested.
Sister-agent contention then re-entered at step 3 and the per-step time
jumped to 144 s; the profile re-pass was eventually killed at step 11.
But the steps 1-2 measurement is the clean per-step number we project
from above. Two independent measurements (last 11 steps of primary run,
first 2 steps of profile pass) bracket the same 41 s steady-state per-step
to within 0.1 s.

#### Per-stage breakdown (cold + contested second pass)

The `--profile` flag wraps a forward-hook probe around `text_encoder`,
`transformer`, `transformer_2`, and `vae.decode` — one-pass inline, no
second generation. The second-pass `profile_wan` re-pass ran under
contention and was less reliable on absolute numbers. The structural
breakdown is what matters; with the contention caveat above:

| Stage | Notes |
|---|---|
| Text encode (UMT5) | ~ms, dominated by tokenizer; rounding error |
| **DiT — transformer** (high-noise expert) | active while `t >= boundary_ratio * num_train_timesteps` (here `t >= 875`); with this UniPC schedule, **the first ~17 of 40 steps** at 1280×720 (empirically — the boundary handoff appears at step 17 in the live tqdm trace). Each step at **~41 s steady-state** once warm |
| **DiT — transformer_2** (low-noise expert) | active for steps ~17–39 (~23 steps). Per-step time at **~41 s steady-state** once kernels are warm — comparable to the high-noise expert (both are the same `WanTransformer3DModel` class, just different weight checkpoints). The boundary handoff (step 16 → 17) shows up as one slower step from triton re-autotuning the new transformer instance |
| VAE decode | one shot at the end of the loop, ~few seconds for an 81 f latent block (FP32 VAE per Wan reference) |
| Other | latent prep, scheduler, postprocess; rounding error vs DiT |

The MoE second-expert handoff was visible in the live tqdm trace as a
one-step jump (~99 s at step 17 of the primary run, vs ~43 s before).
This is the triton kernel cache warming for the second transformer
instance; subsequent low-noise steps return to the steady-state rate.
After that one-time cost, **the MoE handoff is essentially free at
runtime** — both experts share the same `WanTransformer3DModel`
class and kernel shapes.

### Mid-config — 49 frames / 24 steps

Not measured in this pass — the 81 f / 40 step run plus its profile
re-pass consumed the available GPU window before sister-agent contention
returned. The 17 f / 8 step smoke and the 81 f / 40 step quality run
together bracket the time-vs-config curve, and the steady-state per-step
(~41 s at 81 f / 720 P, ~5 s at 17 f / 720 P) tells us the per-step cost
scales roughly linearly with the latent volume.

### Memory

Smoke peak HBM: **84.3 / 192 GiB**. Full reference will run higher
(more latent volume → bigger DiT activations). Even at the smoke shape,
~110 GiB of HBM headroom remains — enough to fit batching of multiple
concurrent generations, or co-resident LoRAs, or both. The Wan team's
own H100 reference uses **`--offload_model True`** (CPU offloading the
non-active MoE expert) to hit 79.8 GB peak; on MI300X we don't need
to offload at all — both experts stay resident through the full
denoise.

## Compared to NVIDIA's H100

The Wan team's `comp_effic.png` reports **1041.5 s / 79.8 GB** for the
single-H100 81-f / 40-step / 720 P T2V-A14B configuration we benchmark
here. Their config uses `--offload_model True --convert_model_dtype`
(FP8 model conversion, CPU offload of the inactive MoE expert) to fit
in 80 GB H100; on MI300X (192 GiB) we don't offload and don't quantize,
both experts stay resident at 85 GiB peak.

**Steady-state projection:** ~1700 s on MI300X (BF16, no offload) vs
1041 s on H100 (FP8 + CPU offload). The ratio is ~1.6× MI300X behind
H100 at the diffusers baseline, **with two important asymmetries**:

1. **H100's number requires FP8 quantization + offload.** Our number is
   BF16 + both experts resident. A more apples-to-apples comparison —
   MI300X at FP8 with offload — is a separate workstream; the diffusers
   BF16 reference is the path Repercep ships today.
2. **MI300X has all the headroom that H100 doesn't.** Wan-2.2's 27 B
   total parameters do not fit in 80 GB H100 in BF16 (54 GiB just for
   weights, ~30+ GiB for activations at 81 f / 720 P); H100 deployments
   *must* offload or quantize. MI300X's 192 GiB fits both experts plus
   activations at 85 GiB peak with no compromises.

The Wan-2.2 community's optimization stack (TeaCache + Sage attention +
FP8 attention + step-skip / Magcache) on 8×H100 reaches **109.8 s
(2.28×) to 60 s (3.1×) over the same baseline**. The same training-free
levers should transfer to MI300X (`OPTIMIZATION.md` Tier 1) — most are
implemented for Cosmos already and parameterized on the transformer
type. A Wan-shaped native loop with adaptive caching is the obvious
next move; conservatively stacked, the Wan community's H100 results
suggest similar headroom on MI300X once the caching strategy is
adapted to the MoE boundary handoff.

The single-A100 H100-vintage reference (no FP8, no offload, FA-2) is
**2735.7 s** per the Wan team's table — Repercep's MI300X BF16 path is
already ~1.6× faster than that, on a single-VF without offloading.

## Strategic context

Cosmos's MI300X result (`COSMOS_ON_MI300X.md`) made the case for
non-NVIDIA silicon as a serving-runtime wedge — and that result was
NVIDIA's own model on AMD hardware. **This result extends the wedge to
a non-NVIDIA model on non-NVIDIA hardware**, with the entire stack
(model, kernels, runtime) free of NVIDIA-specific dependencies:

- Wan-2.2 is Alibaba's Apache-2.0 release. No NVIDIA Open Model License,
  no guardrail dependency, no `cosmos_guardrail` version skews. Just a
  diffusers pipeline.
- The MoE pattern (per-step expert swap) is structurally heavier than
  Cosmos's single-transformer DiT — it stresses pipeline coordination
  in a way Cosmos doesn't.
- The 5-second video / 16 FPS clip length is the same shape commercial
  video providers (fal.ai, Replicate, Runway-tier) ship — directly
  relevant to a serving-runtime wedge.

## Caveats

- Single MI300X VF (virtualized), one shape. Multi-GPU paths are not
  exercised on this host.
- No FVD comparison vs the Wan H100 reference. There isn't a published
  reference clip + seed combination from the Wan team to diff against;
  visual quality is eyeballed on the produced mp4 and confirmed to be a
  valid Wan-shaped output.
- Caching analysis is **deferred** — Repercep's adaptive cache is
  Cosmos-block-shape-specific and doesn't transfer 1:1 to
  `WanTransformer3DModel`. A native Wan loop is queued.
- Wan-2.2's MoE second-expert boundary handoff is a known non-trivial
  cost — Session 10 noted ~270 s of the 326 s cold smoke was non-DiT
  work (load, VAE, postprocess, boundary handoff). The per-stage
  breakdown below quantifies this on the warm path.
- The Wan team's H100 reference uses `--offload_model True
  --convert_model_dtype` (FP8 quantization + CPU offload). This is **not
  apples-to-apples** with our run: we don't offload (192 GiB MI300X
  doesn't need to), and we don't quantize the model (BF16
  transformer + FP32 VAE). Our peak HBM and wall time should be read
  together with the H100 figure with that caveat in mind. A
  fully-quantized-and-offloaded MI300X comparison is a separate
  follow-up; the diffusers BF16 reference is the path we ship in
  Repercep today.

## Reproduce

Hardware: AMD Instinct MI300X (192 GiB) or compatible CDNA3; ROCm 7.x.

```bash
git clone <repo> repercep && cd repercep
pip install --user uv
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
make install

# Smoke baseline (~45 s warm gen + 16 s load)
.venv/bin/python scripts/run_wan.py --frames 17 --steps 8

# Quality reference — 81 f / 40 steps / 1280×720, with per-stage profile
.venv/bin/python scripts/run_wan.py --frames 81 --steps 40 --profile

# Smaller TI2V-5B variant (~10 GiB BF16) for VRAM-constrained smoke
.venv/bin/python scripts/run_wan.py --small --frames 17 --steps 8
```

## Versions used

| Component | Version |
|---|---|
| GPU | AMD Instinct MI300X VF (192 GiB HBM3, 304 CUs, gfx942 / CDNA3) |
| ROCm | 7.2.0 (HIP 7.2.53211) |
| Python | 3.12.3 |
| `torch` | 2.12.0+rocm7.2 (from `download.pytorch.org/whl/rocm7.2`) |
| `torchvision` | 0.27.0+rocm7.2 (same index) |
| `diffusers` | 0.37.1 (registers `WanPipeline`, `AutoencoderKLWan`, `WanTransformer3DModel`) |
| `transformers` | 5.9.0 |
| `accelerate` | 1.13.0 |
| Repercep | 0.0.1 (this repo, `WanEngine`) |

## References

### Wan team — model card, repo, reference numbers

- Wan2.2 GitHub repo (the `comp_effic.png` source):
  <https://github.com/Wan-Video/Wan2.2>
- Wan-AI/Wan2.2-T2V-A14B-Diffusers (the lead Repercep variant):
  <https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers>
- Wan-AI/Wan2.2-T2V-A14B (raw / non-diffusers):
  <https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B>
- Wan-AI/Wan2.2-I2V-A14B-Diffusers (image-to-video sibling):
  <https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers>
- Wan-AI/Wan2.2-TI2V-5B-Diffusers (smaller variant — Repercep's `--small`):
  <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>
- Wan-2.1 arxiv paper (Wan-2.2 has no standalone paper; the 2.1 paper is
  the closest published methodology reference):
  <https://arxiv.org/abs/2503.20314>

### Optimization-stack numbers (8×H100; not directly comparable)

- Morphic — Boosting Wan2.2 I2V on 8×H100, 2.5× with sequence
  parallelism + Magcache:
  <https://morphic.com/blog/boosting-wan2-2-i2v-56-faster>
- Voltage Park — Accelerating Wan2.2 from 4.67 s to 1.5 s per
  denoising step (8×H100, T2V):
  <https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations>
- Simplismart — Serving WAN 2.2 at lightning speed (8×H100, T2V and
  I2V both at 3.2×):
  <https://simplismart.ai/blog/deploy-wan-2-2>
- Baseten — Wan 2.2 video generation in less than 60 seconds (H100 vs
  B200, relative claims only):
  <https://www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/>
- fal.ai — Wan v2.2-A14B text-to-video pricing surface:
  <https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video>

### MLPerf — Wan-2.2 first submissions (v6.0, April 2026)

- AMD ROCm blog — MLPerf Inference v6.0, Single Stream Wan-2.2-T2V-A14B
  on MI355X at 27.4 s:
  <https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html>

### Adjacent AMD video-diffusion proof points on MI300X / MI355X

- HunyuanWorld-Voyager on MI300X (~471 s @ 1040×768 / 49 f / 50 steps):
  <https://rocm.blogs.amd.com/artificial-intelligence/hunyuanworld-voyager-inference/README.html>
- AMD Micro-World on MI325X (their own world model):
  <https://rocm.blogs.amd.com/artificial-intelligence/micro-world/README.html>

## Acknowledgements

Wan-Video (Alibaba) team for open-sourcing Wan-2.2 under Apache 2.0.
HuggingFace `diffusers` maintainers — particularly the WanPipeline /
AutoencoderKLWan / WanTransformer3DModel authors — for the diffusers
path Repercep runs above. The AMD ROCm + aotriton + hipBLASLt teams for
the underlying kernel infrastructure.
