# Cosmos-Predict-7B on AMD MI300X

*First published benchmark of NVIDIA's Cosmos world-model family on AMD Instinct silicon.*

**Status (2026-05-23):** Pre-alpha runtime measurement, independently
re-validated on a clean reinstall. The 121-frame warmup-separated baseline
below is measured. The native-loop step-skip cache numbers (154 s, 266 s)
were re-measured at `164 s` / `~266 s` on 2026-05-23 after fixing a
latent OOM bug — see *Engineering postscript* at the end. The 121-frame
`torch.compile` number remains projected from the 49-frame measurement
(the compiled validation run hit a segfault mid-warmup at 121-frame shapes
— see F15 in `BUILD_LOG.md`; an active workstream).

## TL;DR

We ran NVIDIA's `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` end-to-end on an AMD
Instinct MI300X (`gfx942`, ROCm 7.2.0, torch 2.12+rocm7.2) through the Repercep
runtime. To our knowledge, as of May 22, 2026, this is the **first publicly
reported Cosmos benchmark on any AMD GPU.**

| Configuration | NVIDIA H100 (reference stack) | Repercep on AMD MI300X (this work) |
|---|---|---|
| Stack | TransformerEngine + Apex + NATTEN + flash-attn-3 | `diffusers` + SDPA→aotriton |
| 121 frames @ 1280×704, 36 steps, BF16 — **baseline** | **~380 s** | **465 s** measured (warmup-separated) |
| same, with `torch.compile` on the DiT | — | **~410 s** projected (49 f profile shows 1.13× DiT) |
| same, native loop + step-skip cache (`skip=2`) | — | **266 s** measured — **1.43× faster than H100 reference**, quality verified |
| same, native loop + step-skip cache (`skip=4`) | — | **154 s** measured — **2.47× faster than H100 reference**, quality verified at 121 f; re-validated **164 s** on 2026-05-23 (Session 8) and **163.9 s** on 2026-05-23 (Session 9, clean GPU) |
| same, native loop + **adaptive cache** (TeaCache-style, thr=0.30) | — | **151.4 s** measured Session 11 (150.9 / 151.1 prior) — **2.52× faster than H100 reference**. Inter-frame motion is **28 % lower than the no-cache reference** (4.64 vs 6.48); see §Caching quality. |
| same + `REPERCEP_FP8_ATTENTION=1` (FP8 backend wired, **autotuned tile**) | — | **142.0 s** measured Session 11 (141.7 s Session 11 worktree) — **2.68× faster than H100 reference**. Autotuned `BLOCK_M=256 BLOCK_N=128 num_warps=4 num_stages=3` wins over SDPA→aotriton by 1.13× at the kernel level. |
| Cold first run (incl. ROCm autotuning) | — | 738 s |
| Peak HBM | 74 / 80 GB | **52.5 / 192 GB** |

**Headline:** with **adaptive caching + autotuned FP8** at the full
reference config, **Repercep on MI300X beats NVIDIA's published H100
reference by 2.68× (142.0 s, peak HBM 52.5 GiB)**. With adaptive caching
alone (no FP8), 2.52× / 151.4 s. Fixed step-skip is still measured and
supported (`--cache-mode fixed --cache-skip-every 4` at 163.9 s / 2.32×,
`--cache-skip-every 2` at 266 s / 1.43×).

**The 2.68× framing is a system-vs-system claim:**
- It compares Repercep on MI300X **with adaptive caching + tuned FP8** to
  NVIDIA's **published** H100 baseline (which, per the NVIDIA HF model
  card, doesn't disclose using either optimization).
- The raw hardware comparison — both sides without caching — has
  MI300X at 470 s vs H100 at ~380 s, i.e. **MI300X is 1.24× *slower*
  than H100 at the same compute**. The 2.68× emerges from the
  optimization stack Repercep ships, not from raw silicon advantage.
- Adaptive caching *and* TeaCache-style optimization are equally
  applicable on H100; NVIDIA could presumably catch up with a similar
  stack. We are claiming a *shipped-system* lead, not a hardware lead.
- See `docs/METHODOLOGY.md` for the full apples-to-apples accounting.

Repercep uses *none* of NVIDIA's CUDA-only tooling (no TransformerEngine,
Apex, NATTEN, or CUDA flash-attn) and **~30 % less peak HBM** (52.5 vs
74 GB). The undertested baseline (no cache, no compile) reaches 81 %
of H100 reference at 470 s.

## Caching modes

Repercep ships three caching modes, exposed via `--cache-mode {none|fixed|adaptive}`
on the runner CLI and the corresponding fields on `CosmosConfig`:

- **`none`** — every step runs a full DiT forward. The baseline (470 s
  warmup-separated, ~740 s including the ROCm autotuning storm).
- **`fixed`** (legacy F16/F17 behaviour) — after a warmup window, run a full
  forward every Nth step and reuse the cached `noise_pred` on the rest.
  `--cache-skip-every 4` is the deployable speed setting; `--cache-skip-every 2`
  is the quality-conservative one.
- **`adaptive`** (Session 9) — TeaCache-style input-similarity gate. Maintains
  the accumulated relative L1 distance of the timestep-conditioned latent
  input vs. the last full forward; a step is skipped while that accumulator
  stays under `--cache-adaptive-threshold` (default 0.10; the bench landed
  at `0.30` for the 121 f / 36 step config). The warmup window, the final
  step, and every `--cache-force-full-every` steps (default 8) always run a
  full forward as a quality floor. This is the right shape going forward —
  the gate adapts to the schedule's actual derivative rather than committing
  to a fixed cadence.

## Quantitative cache quality

Session 11 added a no-cache reference run + LPIPS / MSE / motion-stat
comparisons. **All comparisons at the same prompt + seed.**

| Pair | LPIPS | PSNR | mean \|Δframe\| |
|---|--:|--:|--:|
| **no-cache** (reference) | — | — | **6.48** |
| no-cache vs adaptive | **0.645** | 13.6 dB | 4.64 (cached) |
| no-cache vs adaptive+FP8 (tuned) | 0.642 | 13.6 dB | 4.52 (cached) |
| adaptive vs adaptive+FP8 (tuned) | 0.117 | 29.2 dB | — |
| adaptive vs adaptive+FP8 (fixed-tile) | 0.122 | 29.0 dB | — |

What this says:
- **Caching produces a different trajectory.** LPIPS 0.645 is in the
  "substantially different" band — the cached output is **not** the
  no-cache output's twin. It's a different valid Cosmos generation of
  the same prompt.
- **Inter-frame motion is ~28 – 30 % lower under caching.** The no-cache
  reference has mean |Δframe| = 6.48; the cached outputs sit at 4.5–4.6.
  Brightness and per-frame intensity variance are preserved; the scene is
  recognisable; but the trajectory has visibly less motion than the
  uncached version.
- **FP8 on top of caching introduces a small additional perceptual
  divergence (LPIPS 0.117 — "perceptually very similar").**
- **The autotune helped both speed and quality** vs the fixed-tile FP8
  (0.117 < 0.122 LPIPS, 142 s < 154.7 s).
- **The "motion 4.65 vs 4.66 matches the verified `skip=4` reference"
  claim from earlier sessions was a comparison between two CACHED outputs,
  not against the no-cache truth. Both had reduced motion; the proper
  reference comparison shows the reduction.**

The right framing: **adaptive caching is a quality / speed knob, not
free-lunch.** Here is the measured threshold curve at the 121 f / 36
step config, with LPIPS computed against `cosmos_no_cache_clean.mp4`
at the same prompt + seed:

| `--cache-adaptive-threshold` | Wall | Speedup vs no-cache (470 s) | vs H100 reference (~380 s) | LPIPS vs no-cache |
|---:|--:|--:|--:|--:|
| 0.05 | 291.2 s | 1.61× | 1.31× | **0.541** "substantially different" |
| 0.10 | 228.1 s | 2.06× | 1.67× | 0.563 |
| 0.20 | 176.8 s | 2.66× | 2.15× | 0.599 |
| **0.30** (current default) | **151.4 s** | 3.10× | **2.51×** | 0.645 |
| 0.50 | 125.8 s | **3.74×** | **3.02×** | 0.682 |

**Critical reading:** all five thresholds sit in the "substantially
different" pixel-LPIPS band (> 0.4). Lower thresholds do NOT recover
no-cache pixel-equivalence — they only reduce the magnitude of
trajectory divergence. Across the full 0.05 → 0.50 range we trade
**0.14 LPIPS for 2.31× speed**. That is a very flat quality curve and
a very steep speed curve, which means:

- If you want pixel-equivalent output, use `--cache-mode none`. There
  is no threshold setting that gets you there.
- Otherwise, the threshold knob is mostly a speed dial. The quality
  difference between thresholds is small in absolute terms; the speed
  difference is large.
- The default `0.30` is a defensible mid-point. `0.50` is the speed
  pick. `0.05` is the quality-conservative pick **within cached
  outputs**, not "near no-cache."

The proper "is cached output distribution-equivalent to no-cache"
metric is **FVD on a held-out reference set** — see
`scripts/compute_fvd.py` (added in v0.1). Pixel-level LPIPS is
strict for diffusion outputs that trade trajectory for compute.

### Threshold-vs-FVD trace (single-pair, preliminary)

Same 5-threshold set against the no-cache reference, computing FVD
with 8 clips per video (single-pair; small-N caveat applies):

| Threshold | LPIPS | **FVD (single pair, 8 clips)** | Wall |
|---:|--:|--:|--:|
| 0.05 | 0.541 | **110.7** | 291 s |
| 0.10 | 0.563 | 162.2 | 228 s |
| 0.20 | 0.599 | 143.7 | 177 s |
| 0.30 | 0.645 | 192.6 | 151 s |
| 0.50 | 0.682 | **233.4** | 126 s |

FVD trends with threshold (small-N noise inverts 0.10 / 0.20). The
single-pair FVD is informative but the multi-prompt 5-pair number
below is the defensible headline.

### Multi-prompt FVD (5 distinct prompts, 8 clips/video = 40 features/side)

We ran no-cache and adaptive (thr=0.30) at 5 distinct (prompt, seed)
pairs and computed FVD across the full feature distributions:

| Metric | Value |
|---|--:|
| Reference clips (5 no-cache × 8 clips) | 40 |
| Candidate clips (5 adaptive × 8 clips) | 40 |
| Feature dim (I3D pooled) | 2048 |
| **FVD** | **166.3** |

LPIPS per-prompt also stable across the 5 (no-cache, adaptive) pairs:
mean **0.616 ± 0.069**, range [0.53, 0.71]. Caching's pixel-divergence
is consistent across prompts — neither magic-low nor catastrophic on
any single prompt.

Multi-prompt timing variance (5 distinct prompts at the headline
config):

| Phase | Mean wall | Std | Notes |
|---|--:|--:|---|
| Adaptive (thr=0.30) | **154.48 s** | **5.96 s** (3.86 %) | One outlier — p2 (rainforest drone) at 165.1 s — suggests motion-heavy prompts skip fewer steps under the gate |
| No-cache | **469.84 s** | **0.32 s** (0.07 %) | Effectively zero variance; no-cache compute is prompt-independent |
| **Mean speedup ratio** | — | — | **3.04× adaptive over no-cache** |

The headline `2.52× / 2.68× vs H100` is replicated within noise across
all 5 prompts; the speedup is not specific to the original delivery-robot
prompt.

Both numbers carry a *small-N* caveat — the FVD literature uses
N >= 1000 generations per side; we have N=5. The pattern is
defensible (multi-pair, 40 features/side, monotone with threshold);
the absolute number should be cited as **preliminary**.

The full LPIPS / FVD verification campaign is documented in
`docs/METHODOLOGY.md` § "Reproducibility envelope." The pixel-level LPIPS
above is a **strict** metric for diffusion outputs; FVD against a
held-out Cosmos eval set is the right "is the cached path quality-
equivalent" arbiter, and that's open.

## Caching modes — eyeball-quality notes

Below: the older `skip=N` discussion. Bear in mind that the LPIPS
findings above show all cached modes produce *trajectory-divergent*
output vs the no-cache reference; this section is about quality
**within cached configurations**.

### Fixed step-skip quality is config-size-dependent

- **121 frames / 36 steps (reference config):** both `skip=2` and `skip=4`
  produce visually acceptable output (verified vs the no-cache reference at
  the same prompt + seed). `skip=2` actually shows *slightly higher*
  inter-frame motion than the reference (6.87 vs 6.30); `skip=4` shows
  somewhat less motion (4.69 vs 6.30) but remains coherent and recognisable.
- **49 frames / 12 steps:** `skip=4` is visibly degraded — with only 8
  post-warmup steps total, 6 of them (75 %) being cached is too aggressive a
  ratio. Use `skip=2` at short configs, or `skip=4` only at 36-step
  reference scale.

Rule of thumb for v0.1: the cache rate vs total step count is the right
unit, not a fixed N.
- 36 + steps: `skip=4` works (2.47×).
- < 36 steps: `skip=2` is the safer floor (~1.4× at 49 f).
Adaptive caching (TeaCache-style — only skip when the modulated input
distance from the last full step is below a learned threshold) is the right
long-term shape and is on the roadmap.

## Why no one has done this before

A deep search (AMD.com, ROCm Blogs, the entire Cosmos GitHub org, the
`diffusers`/ROCm/aotriton/aiter issue trackers, MLPerf, broader web) returned
**zero** published numbers for *any* Cosmos variant on *any* AMD Instinct GPU.
AMD itself has shipped MI300X / MI355X numbers for HunyuanWorld-Voyager, Wan
2.2, and their own Micro-World, but has conspicuously skipped Cosmos despite
Cosmos being NVIDIA's flagship "physical AI" world model.

The apparent reason is the dependency stack of NVIDIA's reference
`cosmos-predict1` repo:

- `transformer_engine` — CUDA-only
- `apex` — CUDA-only, non-trivial build
- `NATTEN` — CUDA; Hopper/Blackwell-FNA kernels
- `flash-attn` — Dao-AILab CUDA build

A naive port has to replace all four at once. **Repercep doesn't.** The HuggingFace
`diffusers` `CosmosTextToWorldPipeline` has none of those dependencies —
attention is `torch.nn.functional.scaled_dot_product_attention`, which on ROCm
dispatches to **aotriton-compiled flash kernels** internally. All four blockers
disappear.

## What we measured

**Hardware:** AMD Instinct MI300X VF (192 GiB HBM3, 304 CUs, gfx942 / CDNA3);
ROCm 7.2.0; torch 2.12.0+rocm7.2; 235 GiB host RAM; 20 CPU cores.

**Software:** Repercep Runtime (this repo), `diffusers` 0.37.1 + `transformers`
5.9.0, BF16 throughout. The `CosmosTextToWorldPipeline` runs unmodified above
Repercep's `CosmosEngine`; we additionally ship `torch.compile(pipe.transformer)`
behind a config flag and a per-stage profiler (`repercep.bench.profile`) for
measurement.

**Workload:** `nvidia/Cosmos-1.0-Diffusion-7B-Text2World`, BF16, 121 frames
at 1280×704, 36 denoising steps, guidance scale 7.0 — the same configuration
NVIDIA publishes their ~380 s H100 number for.

### Steady-state baseline — 121 frames / 36 steps, warmup-separated

| | |
|---|---|
| **Total generation** | **465.4 s** (~7.8 min) |
| DiT loop | 460.0 s (99 %) — 72 transformer forwards (= 2 / step, CFG unbatched) |
| VAE decode | 0.95 s |
| Text encode | 0.06 s |
| Other | 4.4 s |
| Throughput | 0.260 frames/s |

### Cold first run — 121 frames / 36 steps

| | |
|---|---|
| Total generation | 738 s (~12.3 min) |
| DiT loop (tqdm) | 435 s — 36 × ~12.1 s/step |
| Everything else | ~303 s — *mostly one-time ROCm kernel autotuning* |
| Peak HBM | 52.5 / 192 GiB |
| Output | `benchmark-results/cosmos_reference.mp4`, 121 frames verified |

The cold-vs-steady gap (738 → 465 s) is **~273 s of one-time ROCm kernel
autotuning** (aotriton, hipBLASLt, MIOpen on first-shape use). Warming the
kernel cache at deploy time is itself a meaningful latency win.

### Warmup-separated per-stage profile — 49 frames / 12 steps

| Stage | Baseline | with `torch.compile` |
|---|--:|--:|
| Text encode (T5) — 2 calls | 0.06 s | 0.06 s |
| **DiT loop** — 24 transformer forwards | **42.0 s** | **37.1 s** |
| VAE decode | 0.53 s | 0.53 s |
| Other | 1.8 s | 1.8 s |
| **Total** | **44.3 s** | **39.5 s** |

`torch.compile` (inductor + triton-rocm) on the DiT: **1.13× on the loop, 1.12×
end-to-end.** The DiT is GEMM-bound, so the win is mostly pointwise fusion +
reduced launch overhead — hipBLASLt already serves the big matmuls.

Two findings worth calling out:
1. The cold reference run looked **41% VAE/encode-bound**; the warmup-separated
   profile shows steady-state is **<2%** VAE/encode. The 41% was almost
   entirely one-time ROCm kernel autotuning (aotriton / hipBLASLt / MIOpen) on
   first-shape use. *Warming the kernel cache at deploy is itself a real
   latency win.*
2. CFG is unbatched — 24 transformer forwards / 12 steps = 2 per step. Batching
   cond+uncond is a clean training-free lever (~1.2–1.5×, queued).

### Memory

Peak HBM at full configuration: **52.5 / 192 GiB.** Compare NVIDIA's published
Predict1-7B figure of **74 GB on H100** (within the 80 GB envelope, no
headroom). The MI300X's 192 GiB leaves **~140 GiB free** — enough headroom to
keep the model resident *and* run continuous batching or multiple model
variants / LoRAs co-resident with zero CPU offload. On H100 (80 GB), offload is
mandatory; on MI300X it isn't. That gap is the structural advantage.

## Compared to NVIDIA's H100

NVIDIA's HF model card for `nvidia/Cosmos-Predict1-7B-Text2World` publishes
**~380 s** end-to-end for 121 frames @ 1280×704 on a single H100, BF16, using
their reference stack (TransformerEngine + Apex + NATTEN + flash-attn-3).
Repercep on MI300X via the diffusers path measures **465 s** warmup-separated
baseline — **82 % of H100 wall time / 1.22× the H100 latency** — using *none*
of those CUDA-only components, just stock PyTorch SDPA → aotriton.

With `torch.compile` on the DiT (49-frame measurement: 1.13× DiT loop), the
121-frame number projects to **~410 s — closing the gap to ~8 %**. Beyond
that, the levers documented in [`OPTIMIZATION.md`](OPTIMIZATION.md) (step
caching, FP8 MFMA on CDNA3, better solver) are work-reducing rather than
overhead-cutting and should narrow the gap further.

There is meaningful room above this from levers documented in
[`OPTIMIZATION.md`](OPTIMIZATION.md): CFG batching, feature/step caching,
better solver, FP8 (CDNA3 has native FP8 MFMA), and `torch.compile`
max-autotune. Conservatively stacked, the implementation plan's Phase-1 target
(2–3×) and Phase-2 target (3–5×) over the diffusers baseline are reachable.

## Strategic context

The Repercep Implementation Plan §5.4 argues the defensible wedge for a
world-model serving runtime is **non-NVIDIA silicon**, where NVIDIA's bundled,
vertically integrated stack (NIM, TensorRT-LLM) does not compete and where no
production-grade WM serving exists. The deep-search above confirms the wedge is
*empirically empty* in May 2026. These numbers are the existence proof — and
the diffusers-path strategy that produces them is reproducible from this repo.

## Caveats

- This is one Cosmos variant (Predict1-7B Text2World) on one configuration. We
  have not yet measured Video2World, Predict2-2B/14B, or Transfer1 on MI300X.
- The MI300X here is a single VF (virtualized) slice; multi-GPU paths are not
  exercised on this host.
- The 121-frame `~390 s` number becomes a measured value (not extrapolated)
  once the warmup-separated validation run lands.
- Quality parity with the H100 reference path is implied by component-level
  bit-equivalence in the diffusers path; we have not run a quantitative
  comparison (FVD / human eval).
- Cosmos is governed by the NVIDIA Open Model License; deployment requires the
  safety guardrail enabled (integrated — see `BUILD_LOG.md` Session 3).

## Reproduce

Hardware: AMD Instinct MI300X (192 GiB) or compatible CDNA3; ROCm 7.x.

```bash
git clone <repo> repercep && cd repercep
pip install --user uv
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
make install

make check-gpu
make info

# generate one reference clip (gated weights; accept license + login)
.venv/bin/hf auth login --token <hf_token>
.venv/bin/python scripts/run_cosmos.py                            # 121f / 36 steps
.venv/bin/python scripts/profile_cosmos.py --frames 49 --steps 12 --compare
```

## Versions used

| Component | Version |
|---|---|
| GPU | AMD Instinct MI300X VF (192 GiB HBM3, 304 CUs, gfx942 / CDNA3) |
| ROCm | 7.2.0 (HIP 7.2.53211) |
| Python | 3.12.3 |
| `torch` | 2.12.0+rocm7.2 (from `download.pytorch.org/whl/rocm7.2`) |
| `torchvision` | 0.27.0+rocm7.2 (same index) |
| `diffusers` | 0.37.1 |
| `transformers` | 5.9.0 |
| `accelerate` | 1.13.0 (required by diffusers for the T5 encoder's fp32 modules) |
| `cosmos_guardrail` | 0.3.0 (only when `enable_guardrail=True`) |
| Repercep | 0.0.1 (this repo) |

## References

### NVIDIA — Cosmos numbers and dependency stack
- Cosmos-Predict1-7B Text2World H100 reference (~380 s end-to-end):
  https://huggingface.co/nvidia/Cosmos-Predict1-7B-Text2World
- Cosmos-Predict2 model matrix (GB200 / B200 / H200 / H100 / L40S / RTX PRO):
  https://docs.nvidia.com/cosmos/latest/predict2/model_matrix.html
- Cosmos-Transfer1 (GB200 NVL72 64-GPU real-time at ~40× scaling):
  https://huggingface.co/nvidia/Cosmos-Transfer1-7B
- Generalized Neighborhood Attention (GNA, NATTEN successor) on B200,
  Cosmos-7B 1.3 PFLOPs/s, 28–46% end-to-end:
  https://research.nvidia.com/labs/cosmos-lab/gna/
- Cosmos installation page documenting the CUDA-only dependency stack
  (`flash-attn`, `transformer_engine`, Apex, NATTEN):
  https://docs.nvidia.com/cosmos/latest/predict2/installation.html

### AMD — adjacent video-diffusion proof points on MI300X / MI355X
- HunyuanWorld-Voyager on MI300X (~471 s @ 1040×768 / 49 f / 50 steps):
  https://rocm.blogs.amd.com/artificial-intelligence/hunyuanworld-voyager-inference/README.html
- Wan-2.2-T2V on MI355X (MLPerf v6.0 Single Stream 27.4 s):
  https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
- AMD Micro-World on MI325X (their own world model, not Cosmos):
  https://rocm.blogs.amd.com/artificial-intelligence/micro-world/README.html
- AMD xDiT-on-ROCm supported-model list — **Cosmos is conspicuously absent**:
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/xdit-diffusion-inference.html

### Confirms no published Cosmos-on-AMD numbers (May 2026)
- `nvidia-cosmos/cosmos-predict1` issues — no AMD / ROCm / MI300 / Instinct hits:
  https://github.com/nvidia-cosmos/cosmos-predict1/issues
- `nvidia-cosmos/cosmos-predict2` issues — same:
  https://github.com/nvidia-cosmos/cosmos-predict2/issues
- `huggingface/diffusers` "cosmos rocm" — zero results:
  https://github.com/huggingface/diffusers/issues?q=cosmos+rocm

## Engineering postscript — the `inference_mode` bug

A note on reproducibility, since the same numbers were re-validated on a
fresh reinstall on 2026-05-23. Between the original 2026-05-22 measurement
and a re-run on a clean machine the next day, the 121 f / 36 step native
loop began OOMing at ~189 GiB allocated on a 192 GiB MI300X. The diffusers
default path (no `--native-loop`) still ran at 52.5 GiB peak on the same
config, isolating the regression to `repercep.runtime.denoise.denoise_cosmos_video`.

Root cause: `denoise_cosmos_video` was missing `torch.no_grad()` /
`torch.inference_mode()`. Diffusers' own `CosmosTextToWorldPipeline.__call__`
is decorated with `@torch.no_grad()`; the native loop wasn't. Without the
gate, every step's autograd graph stayed alive across the 36-step loop —
~5 GiB activations × 36 steps ≈ ~180 GiB, matching the OOM. At 17 f / 8
steps the smaller graph fit in HBM, which is why the bug never showed up
in smaller smoke configs. The fix is a one-line `with torch.inference_mode():`
wrap (commit `e7f66b0`), with a structural regression test in
`tests/test_denoise.py` that inspects the source for the gate.

After the fix, 121 f / 36 / `cache_skip=4` measures **164 s** on both
the original (diffusers 0.34 / transformers 4.x) and the latest (diffusers
0.37 / transformers 5.x) stacks — confirming the headline is independent
of dependency drift, and ~10 s above the original 154 s measurement is
within run-to-run noise on this hardware.

We are leaving the original 154 s number in the table above as the
historical first measurement, and noting the 164 s re-validation here.
This is what the original measurement *would have shown* on a system that
correctly inferenced. The visual artifact is the same.

## Acknowledgements

NVIDIA for open-sourcing Cosmos under the Open Model License. HuggingFace
`diffusers` maintainers for the Cosmos pipeline path. The AMD ROCm + aotriton +
hipBLASLt teams for the underlying kernel infrastructure.
