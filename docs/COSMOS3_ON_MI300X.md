# Cosmos 3 Nano Policy on MI300X — reference-stack numbers (2026-08-09)

**The port's headline question, answered: Cosmos 3 Nano's policy path runs on
AMD.** No CUDA-only dependency, no stubbing, no source patches — diffusers'
`Cosmos3OmniPipeline` on stock ROCm PyTorch, first try.

These are **reference-stack** numbers — the diffusers pipeline called directly,
*not* through the Repercep seam (`models/cosmos3.py`'s pipeline half is still
Phase 1). That is why there is no row in `CONTROL_LOOP_BENCH.md`: putting one
there would claim a provenance this run does not have. Raw lines:
`docs/results/cosmos3_nano_policy_mi300x_2026-08-09.json`. Plan and the
pre-registered gates: `docs/COSMOS3_PORT_PLAN.md`.

## 1. Environment

| | |
|---|---|
| GPU | AMD Instinct MI300X VF (gfx942), 191.7 GiB HBM |
| Host | AMD Developer Cloud (DigitalOcean-backed), `gpu-mi300x1-192gb-devcloud`, `atl1`, $1.99/hr |
| ROCm | **7.0.2**, torch **2.10.0+rocm7.0** (HIP 7.0.51831), Python 3.12.3 |
| diffusers | **0.40.0.dev0 @ `d6726f38`** (main — no released wheel carries `Cosmos3OmniPipeline`) |
| transformers | 5.14.1 |
| Model | `nvidia/Cosmos3-Nano-Policy-DROID`, bf16, **public and ungated** (32.94 GiB) |
| Guardrail | disabled (`enable_safety_checker=False`) |

Setup deviates from `amd-developer-cloud-access`'s recipe in one way worth
keeping: **a plain venv with torch installed into it**, rather than a
system-level torch plus `venv --system-site-packages`. That sidesteps the
RECORD-less Debian `typing_extensions` failure entirely. `apt install
python3.12-venv` is still required first, and torch still must come from
`--index-url https://download.pytorch.org/whl/rocm7.0`.

## 2. The ROCm gate — passed, and cheaply

`scripts/cosmos3_rocm_probe.py` stages the checks by cost so the 33 GB download
is last. Stage 2 is the one the port rested on:

```
[PASS] stage 0: torch + device -- AMD Instinct MI300X VF, torch 2.10.0+rocm7.0, ROCm 7.0.51831
[PASS] stage 1: diffusers + Cosmos 3 symbols -- diffusers 0.40.0.dev0
[PASS] stage 2: CUDA-only dependency check -- 0 modules pulled, no CUDA-only imports
       pipeline module: diffusers.pipelines.cosmos.pipeline_cosmos3_omni
[PASS] stage 3: construct from config -- Cosmos3OmniTransformer, 15.17B params (meta)
```

Zero `transformer_engine` / `apex` / `flashinfer` imports, and no `flash_attn`
either — so unlike DreamZero there was not even a guarded fallback to rely on.
The port plan's §6 risk ("if the pipeline pulls TransformerEngine the AMD claim
is gone") did not fire.

MIOpen logs `Error [Init] Not found :<N>-DeviceGroupedConvFwd...` repeatedly
during the first calls. These are **kernel-selection misses that fall back
successfully**, not failures — output is correct and steady-state timing is
stable. They are the visible edge of the cold-start tax in §4.

## 3. Architecture facts, read off the shipped config

Worth recording because most are absent from the model card and two correct
guesses made in the port plan:

| | |
|---|---|
| transformer | `Cosmos3OmniTransformer`, **15.17B params** (the "16B" headline includes VAE + vision encoder) |
| reasoner backbone | **`model_type: qwen3_vl_text`** — the reasoning tower is Qwen3-VL-derived |
| MoE | **`use_moe: true`** |
| VAE | **`AutoencoderKLWan`** — the *Wan* VAE, the same family `models/wan.py` already serves |
| vision encoder | `Qwen3VLVisionModel` |
| attention | `joint_attn_implementation: two_way`, `qk_norm_for_diffusion: true` |
| positions | `unified_3d_mrope`, `max_position_embeddings: 262144` |
| dims | hidden 4096, intermediate 12288, head_dim 128, latent_channel 48, latent_patch_size 2 |
| **action register** | **`action_dim: 64` / `max_action_dim: 64` (padded)** — DROID's *used* width is 10 |

**The action-width correction matters.** `Cosmos3Config.action_dim = 10` in the
scaffold is the *wire* width; the model's register is **64-wide, zero-padded**
across embodiments — exactly the pattern DreamZero has (32 padded / 8 used) and
which its config models as `action_dim` + `used_action_dim`. Phase 1 should
mirror that split rather than carry one number. The pipeline returned `(16, 10)`,
confirming it de-pads on the way out, so the scaffold is not *wrong* today — but
it is under-described, and a post-trained checkpoint on another embodiment would
expose that.

## 4. Latency

Steady state, 16-action policy chunk at 480p / 30 denoise steps / `flow_shift=5.0`:

| | ms | note |
|---|---:|---|
| **chunk latency (steady state)** | **3610** | median of 7 samples, spread **±0.3%** |
| ├ VAE decode (`AutoencoderKLWan`) | 452 | 12.5% of the chunk |
| └ encode + 30 denoise steps | 3160 | |
| peak HBM | 33.59 GiB | resident weights 29.7 GiB |
| weight load | 82.1 s | from local disk cache |

### 4.1 The cold-start tax is the largest this repo has measured

| call | ms | vs steady state |
|---|---:|---:|
| first call ever on a fresh box (cold MIOpen disk cache) | **253,159** | **70×** |
| first call in a fresh process (warm MIOpen cache) | 15,454 | 4.3× |
| steady state | 3,610 | 1× |

DreamZero's MI300X row recorded a 49.6 s first call (~8.4× warm) and called it
"larger than the ~3× rule of thumb elsewhere in this repo's ROCm numbers." This
is four minutes, and it is **70×**. Two distinct effects, now separated: MIOpen
autotune results persist on disk, so the catastrophic case is once per *machine
image*, while the 4.3× is once per *process*.

**This is an operational finding, not a benchmark artifact, and it is the most
directly Repercep-relevant thing in this run.** An inference cloud that
cold-starts a Cosmos 3 worker onto fresh ROCm capacity pays four minutes before
the first token of useful work. Any autoscaling story on AMD has to ship a
pre-warmed MIOpen cache in the image, or it does not work. That belongs in the
deployment playbook.

### 4.2 Real-time framing — misses at defaults, **makes it on the ladder**

A 16-action chunk at 15 FPS covers **1.067 s** of robot time. At the reference
defaults it takes 3.61 s to produce: **3.4× slower than real time**, 4.43
actions/s against 15 required. At *those* settings the model is a
simulation/eval engine, not a closed-loop controller.

The ladder in §4.4 changes that conclusion: at **256p / 5 steps the chunk takes
479.5 ms — 2.2× faster than real time.** Whether the resulting trajectories are
good enough to close a loop with is a separate, unanswered question (§4.5).

### 4.3 The cold-start tax is per-*shape*, not just per-process

The first 256p run measured 49.8 s and looked like evidence that the smaller
tier was catastrophically slower. It was not — it was autotune for a shape
MIOpen had never seen. Repeated three times:

| tier / steps | call 1 | call 2 | call 3 |
|---|---:|---:|---:|
| 256p / 30 | **13,820.7** | 1,831.5 | 1,834.8 |
| 256p / 5 | 480.1 | 479.5 | 476.4 |
| 480p / 30 | 3,645.1 | 3,610.2 | 3,611.8 |
| 480p / 5 | 1,164.8 | 1,161.6 | 1,156.5 |

**7.5× on the first call at a new resolution**, then flat. The 480p rows are
flat from call 1 because that shape had already been warmed. Changing
`num_inference_steps` costs nothing extra — it changes iteration count, not
tensor shapes — which is why the denoise ladder shows no such penalty.

This sharpens §4.1 into something more operationally demanding: **a serving
fleet must pre-warm every (resolution, batch) shape it intends to serve**, not
merely load weights once. Pre-warming the weights and one shape still leaves a
7.5× cliff on the first request at any other resolution.

### 4.4 Levers ladder (warm, MI300X)

All rows vs the 480p/30-step reference at 3611.8 ms. Real-time budget is
1067 ms.

| tier | steps | ms | speedup | vs real time | action drift (max abs / mean rel) |
|---|---:|---:|---:|---:|---|
| 480 | 30 | 3611.8 | 1.00× | 3.39× slower | — (reference) |
| 480 | 20 | 2628.6 | 1.37× | 2.46× slower | 0.055 / 1.9% |
| 480 | 15 | 2134.5 | 1.69× | 2.00× slower | 0.078 / 3.3% |
| 480 | 10 | 1648.8 | 2.19× | 1.55× slower | 0.109 / 5.1% |
| 480 | 5 | 1161.6 | 3.11× | 1.09× slower | 0.188 / 9.1% |
| 256 | 30 | 1834.8 | **1.97×** | 1.72× slower | 0.098 / 6.6% |
| 256 | 5 | **479.5** | **7.53×** | **2.23× faster** | not measured |

Drift is the predicted action chunk's deviation from the 30-step/480p baseline,
in the model's normalized `[-1, 1]` space — `max abs` over all 160 values and
`mean abs` relative to the baseline's mean magnitude. It is a **cheap proxy for
"how much does cutting compute change the decision,"** not a quality metric: it
says nothing about which trajectory is *better*, only how far they diverge.

The denoise-step rows are single samples (the repeated runs in §4.3 cover only
the 30- and 5-step ends, where run-to-run spread was ≤0.7%). The two ends are
solid; the middle is indicative.

**Resolution is the better lever than denoise steps.** 256p/30 buys 1.97× at
6.6% drift, while 480p/10 buys a comparable 2.19× at 5.1% — but combining them
is where it gets interesting, and 256p/5 at 479.5 ms is a 7.5× total that lands
comfortably inside the real-time budget.

### 4.5 What the ladder does not establish

**Whether any of these settings are usable.** Drift of 9% in normalized action
space could be irrelevant or disqualifying depending on the task, and this run
cannot tell which, for two compounding reasons: the conditioning was a flat grey
canvas (§6), so the baseline trajectory is itself meaningless; and there is no
success-rate harness attached. NVIDIA ships one — `RoboLab` — and running the
`BananaInBowlTask` suite against each rung is what would turn this ladder from
a latency table into a claim. That is the natural next piece of GPU work, and
until it exists **no rung below 30 steps should be quoted as a serving
configuration.**

### 4.6 Against the other DROID model on this GPU

| | Cosmos3-Nano-Policy-DROID | DreamZero-DROID |
|---|---|---|
| chunk latency (MI300X) | **3610 ms** | 5646.6 ms |
| actions/chunk | 16 | 24 |
| **ms/action** | **226** | 235 |
| weights resident | 29.7 GiB | 42.78 GiB |
| per-session marginal HBM | ~0 (stateless) | 23.05 GiB |
| resident sessions/GPU | **weights-bound, not state-bound** | 6 |

**Read this as serving cost on one robot, never as a policy comparison.** Same
DROID hardware, different action representations — Cosmos 3 emits 10D
end-effector pose deltas, DreamZero 8D joint positions (port plan §1.1).

The structurally interesting column is the last two. DreamZero pays 23 GiB of KV
*per session*, capping an H100 at one session and an MI300X at six. Cosmos 3 is
stateless per chunk (port plan §2.1), so a session costs one RGB frame and **one
29.7 GiB weight copy serves all of them** — concurrency is bounded by batching
and compute, not by memory per session. On a 191.7 GiB MI300X that is a much
better shape.

## 5. The pre-registered batching gate — and the fake number it nearly produced

Port plan §5 pre-registered this before any measurement, predicting
prefix-sharing near 1× and warning that any win would be plain batch efficiency,
which is stock and therefore playbook rather than moat.

**Actual verdict: not measurable — the stock interface exposes no
candidate-batching axis at all.** `Cosmos3OmniPipeline.__call__` has no
`num_videos_per_prompt` / `num_images_per_prompt` parameter, and
`CosmosActionCondition` carries a single `image` that drives the latent batch.

Passing `prompt=[p]*N` produced, at N = 1, 2, 4, 8:

- **1** candidate returned every time,
- wall time **3595–3610 ms** — statistically identical to N=1,
- peak HBM **33.59 GiB** — identical to N=1.

The batch was silently ignored. **A naive `batched_ms / N` reading of that run
gives 2.01× / 4.00× / 8.01×, and it is entirely fake** — an unchanged runtime
divided by N. It is recorded here, and in the results JSON, specifically so it
is never re-derived by someone reading the raw lines and mistaken for a result.
Constant wall time *and* constant peak memory across a 8× batch sweep is the
tell; either alone might have been explained away.

**What this changes strategically, stated carefully.** In July the best-of-N
lever died because `n=N` was stock in every LLM engine — we had the lever but
could not capture it. Here the opposite holds: candidate fan-out is **not**
available through the stock interface, so a customer cannot get it from anyone
by passing a flag, and implementing it would genuinely be ours. That is a
*reason to build*, not a result. It requires modifying the pipeline's latent
preparation, and **whether it then yields a win is still completely unmeasured**
— Phase 2, with the sequential baseline (8 × 3.61 s = 28.9 s for eight
candidates) as the number to beat. Nothing about batching goes in any external
artifact until that is done.

### 4.7 H100 comparison (added 2026-08-09, same day, same script)

Run on a RunPod H100 80GB HBM3 with the same pinned diffusers `d6726f38`, same
bf16, same policy geometry. Same `warm.py`, so the numbers are directly
comparable.

| | MI300X | H100 | MI300X / H100 |
|---|---:|---:|---:|
| **chunk latency (warm median)** | 3610 ms | **3013 ms** | **0.83×** |
| VAE decode | 452 ms | 345 ms | 0.76× |
| resident weights | 29.70 GiB | 29.66 GiB | 1.00× |
| peak HBM | 33.59 GiB | 33.42 GiB | 1.00× |
| **first call in a fresh process** | 15,454 ms (**4.28×** steady) | 3,824 ms (**1.27×** steady) | — |

Three readings, in descending confidence:

1. **H100 is ~1.2× faster on this workload.** MI300X lands at 0.83× — above the
   0.75 line the project uses as a kill threshold for MI300X, though that
   threshold was written for *token* throughput and this is diffusion. It is a
   data point, not the gate (`LLM_SILICON_GATE_RESULT.md`).
2. **Identical memory to two decimal places** (29.70 vs 29.66 GiB resident,
   33.59 vs 33.42 GiB peak) — the same model doing the same work, which is a
   good sign the two runs are genuinely comparable rather than accidentally
   configured apart.
3. **The cold-start gap is the real vendor difference: 4.28× on ROCm against
   1.27× on CUDA.** Steady-state throughput differs by 20%; *first-call* cost
   differs by 4×. On the whole-machine cold path it is far starker still — 253 s
   on a fresh MI300X box against 7.5 s for the H100's first call. Combined with
   §4.3's per-shape finding, cold start is where AMD actually loses on this
   workload, and it is a solvable engineering problem (pre-warmed caches) rather
   than a silicon deficit.

Weight-load time is **not** compared: H100 showed 7.7 s against MI300X's 82.1 s,
but the two boxes had different storage (RunPod network volume vs local disk) and
different page-cache states. That number is provider noise, not a vendor result.

## 5b. `forward_dynamics` works on the Policy-DROID checkpoint — the seam is safe

The port plan flagged this as a Phase-1 gate: the fd path is demonstrated on the
base `Cosmos3-Nano` while policy is demonstrated on `Cosmos3-Nano-Policy-DROID`,
and if the post-trained checkpoint had lost fd then `models/cosmos3.py`'s
`step()` would have had to fall back to policy mode and **stop being
action-conditioned at all**.

It works. Feeding a policy call's own predicted `(16, 10)` chunk back as
`raw_actions` in `forward_dynamics` mode: **3565.1 ms**, 17 frames, and
`result.action is None`.

Three things settled at once: `step()` can be genuinely action-conditioned, so
`Cosmos3Config.step_mode` keeps its `forward_dynamics` default; fd costs
essentially the same as policy (3565 vs 3612 ms), so the seam's two verbs are
symmetric in price; and the `_Cosmos3Pipeline` Protocol's documented contract —
*"`actions_norm` … is `None` in forward-dynamics mode"* — is confirmed against
the real pipeline rather than inferred from the notebooks.

## 6. Caveats on these numbers

- **Synthetic conditioning.** A flat 640×540 grey canvas, not the real
  three-camera DROID composite. Latency is dominated by fixed-size denoise work
  so the timing should hold, but the *outputs* are meaningless and nothing about
  trajectory quality can be read off this run.
- **One machine, one session.** No multi-seed averaging across boxes, no
  concurrency curve. The ±0.3% spread is within-process repeatability, not
  run-to-run variance.
- **No H100 row yet**, so every cross-vendor statement here is deferred rather
  than made. MI300X went first deliberately (port plan §6) because ROCm was the
  gate the port depended on.
- **Guardrail disabled.** Fine for latency; a deployment claim would need it on
  and re-measured, since it adds a text check before and a video check after.
- **Drift ≠ quality** (§4.5). No success-rate harness was run; `RoboLab` is the
  missing piece before any reduced-compute rung can be called a serving config.

## 7. What this run settles, and what it does not

**Settles:** Cosmos 3 Nano's policy path runs clean on MI300X/ROCm with no
patches. 3.61 s/chunk at reference defaults, 226 ms/action, 29.7 GiB resident,
stateless sessions. `forward_dynamics` survives post-training, so the seam's
`step()` stays action-conditioned. A cold-start tax that is **per-shape** (7.5×)
as well as per-process (4.3×) and per-machine-image (70×) — the single most
actionable operational finding here. And, on the ladder, **real-time-capable
DROID policy inference on AMD: 479.5 ms/chunk at 256p/5 steps against a 1067 ms
budget.** The AMD claim in the port plan is **earned** — as of this run nobody
else has published a Cosmos 3 ROCm number, and NVIDIA's own
`inference_benchmarks.md` still has no policy row on any hardware.

**Does not settle:** anything comparative against H100; whether any reduced rung
is *usable* (drift is not quality — §4.5, and `RoboLab` is the missing harness);
anything about candidate batching beyond "the stock interface cannot do it";
anything about behaviour on real camera input rather than a grey canvas.
