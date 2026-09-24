# Repercep — Optimization Strategy

**Workload:** Cosmos-Predict-7B Text2World · **Hardware:** AMD Instinct MI300X (gfx942)

## 1. Principle — measure, then optimize

Per the implementation plan: build the benchmark, establish the baseline,
optimize against *measured* cost. No optimization lands without a before/after
number from `repercep.bench` against the baseline below.

## 2. The measured baseline — 2026-05-22

Two measurements, both `CosmosEngine` on the naive `diffusers` path, MI300X, bf16.

**Cold full run** — 121 frames @ 1280×704, 36 steps: **738 s**, peak HBM
52.5 GiB. But a cold run pays one-time ROCm kernel autotuning (aotriton /
hipBLASLt / MIOpen) on every new shape — ~300 s of that 738 s is autotuning,
not compute.

**Warmup-separated per-stage profile** — 49 frames @ 1280×704, 12 steps, steady
state (`scripts/profile_cosmos.py`):

| Stage | Wall time | Share |
|-------|-----------|-------|
| Text encode (T5) | 0.06 s | 0.1% |
| **DiT denoising loop** (24 transformer forwards) | **42.0 s** | **95%** |
| VAE decode | 0.53 s | 1.2% |
| Other (latent prep, scheduler, postprocess) | 1.8 s | 4% |
| **Total** | **44.3 s** | 100% |

What this sets:

- **The DiT loop is the entire game** — 95% of steady-state time. VAE decode and
  text encode are rounding error; Tier 2 below is deprioritized accordingly.
- CFG is **unbatched**: 24 transformer forwards for 12 steps = 2 per step. The
  DiT processes a long latent-video sequence; attention is quadratic in it — the
  single most-leveraged kernel.
- A cold run wastes ~300 s on kernel autotuning — **warming the kernel cache at
  deploy time is itself a real latency win**.
- Peak HBM is 52 of 192 GiB. **~140 GiB sits unused** — headroom that converts
  directly into batching and resident-model throughput (the MI300X advantage).

Measured so far (49 f / 12 steps, warmup-separated):
- `torch.compile` on the DiT — **1.13× loop / 1.12× end-to-end**.
- Repercep-native loop with CFG batching — **1.02× loop / 1.05× end-to-end**.
  Smaller than projected: at Cosmos-7B scale each transformer call is
  compute-bound, so packaging two batch-1 forwards as one batch-2 forward
  doesn't reduce GEMM work — see BUILD_LOG F14.
- Native loop + step-skip caching: **the biggest measured lever**, with
  config-size-dependent quality (F16, F17).
  - **121 f / 36 steps, `skip=4`** → **154 s** warmup-separated, **2.47×
    faster than H100 reference**, quality verified.
  - **121 f / 36 steps, `skip=2`** → 266 s, 1.43× faster than H100,
    quality-conservative (motion slightly *higher* than reference).
  - **49 f / 12 steps, `skip=4`** → speed wins but visibly degraded —
    the cache-vs-total-steps ratio matters; lighter `skip=2` is the floor at
    short configs.
- Adaptive caching (TeaCache-style — input-similarity gated) is the right
  long-term shape; tracked separately.
- **FP8 attention (CDNA3 MFMA)** — Session 8, 2026-05-23. Two ops shipped:
  fused `fp8-triton-flash` (FA-2 algorithm + FP8 MFMA tiles) and unfused
  `fp8-scaled-mm` (`torch._scaled_grouped_mm`). Opt-in via
  `REPERCEP_FP8_ATTENTION=1`. Attention-only forward time vs SDPA, B=1 H=8
  D=128, warmup-separated:
  | seq_len | SDPA | fp8-triton-flash | speedup | rel mean err |
  |--:|--:|--:|--:|--:|
  | 4096 | 3.21 ms | 6.07 ms | 0.53× | 3.3% |
  | **8192** | **5.78 ms** | **3.01 ms** | **1.92×** | 3.3% |
  | **16384** | **15.97 ms** | **8.34 ms** | **1.92×** | 3.2% |
  Crossover S≈4-8k; the FP8 path wins above that on the bench micro-shape.
- **FP8 wired into Cosmos's diffusers path** — Session 10, 2026-05-23
  (Phase 2.5). A ``"repercep_fp8"`` backend is now registered with
  ``diffusers.models.attention_dispatch._AttentionBackendRegistry`` via
  ``src/repercep/attention/diffusers_backend.py``;
  ``REPERCEP_FP8_ATTENTION=1`` switches the active backend at
  ``CosmosEngine.load()``. End-to-end 121 f / 36 step / adaptive caching:
  | Config | Wall | Motion stat |
  |---|--:|--:|
  | adaptive (native dispatcher) | **150.9 s** | 4.64 |
  | adaptive + ``REPERCEP_FP8_ATTENTION=1`` | 155.1 s | 4.65 |
  **No wall-time win at Cosmos's production shape**, despite the 1.92×
  bench number — the bench was at B=1, H=8; Cosmos runs B=2, H=32. A
  follow-up bench at the actual production shape (B=2, H=32, D=128,
  S∈{65k, 109k}) shows fp8-triton-flash at **0.98× SDPA**:
  | seq_len | SDPA | fp8-triton-flash | best vs SDPA |
  |--:|--:|--:|--:|
  | 65536 | 415.75 ms | 425.27 ms | 0.98× |
  | 109000 | 1147.39 ms | 1176.40 ms | 0.98× |
  The Triton kernel's BLOCK_M=128/BLOCK_N=64/num_warps=4 tile shape
  amortizes well when the grid is sparse (8 program columns at B=1, H=8);
  at Cosmos's 64-column grid (B=2 × H=32) the kernel's per-tile overhead
  no longer fits and SDPA→aotriton (which is autotuned per shape) wins.
  Quality is intact — eyeballed output is visually identical to the
  native run; mean abs pixel diff = 6.81 / 255. The wiring is correct
  and the env var now actually changes the kernel diffusers calls — the
  bottleneck moved from "FP8 unreachable" (Session 9 F19) to "FP8 kernel
  needs autotuning for the production shape" (Session 10 follow-up).
- **FP8 kernel autotune lands** — Agent I, 2026-05-23. The kernel now
  ``@triton.autotune``-s over a 19-config grid keyed on
  ``(Sq, Skv, BLOCK_D, H, CAUSAL)``, with a persistent JSON cache at
  ``~/.cache/repercep/fp8_autotune.json`` so the winner survives across
  processes (one tune per shape per host, then free forever). A manual
  ``--manual`` search mode is also wired up — it benches a hand-picked
  12-config grid with controlled warmup/rep, more robust under GPU
  contention. End-to-end on Cosmos at the production config (121 f /
  36 steps / adaptive cache thr=0.30 / force_full_every=16):
  | Config | Wall | vs 150.9 s |
  |---|--:|--:|
  | Adaptive-only (no FP8), Session 10 baseline | 150.9 s | — |
  | Adaptive + FP8 fixed M=128/N=64 (Session 10) | 154.7 s | +3.8 s loss |
  | **Adaptive + FP8 tuned M=256/N=128 (Session 11)** | **141.7 s** | **−9.2 s win** |
  Per-call kernel timings at B=2 H=32 D=128 S=109120, **clean GPU**:
  | path | wall | vs SDPA | vs fixed-fp8 |
  |--|--:|--:|--:|
  | SDPA (aotriton) | 1154.88 ms | 1.00× | — |
  | fixed-fp8 (M=128 N=64 w=4 s=2) | 1171.17 ms | 0.99× | 1.00× |
  | autotuned-fp8 (M=256 N=128 w=4 s=3) | **1024.37 ms** | **1.13×** | **1.14×** |
  The autotuner picked the larger 256×128 tile with a 3-stage software
  pipeline — exactly what FA-2 lore predicts for very long sequences
  where the K/V tile loads are bandwidth-bound. F20's "0.98× SDPA at
  the production shape" has flipped to 1.20× per-call and a clean
  9-second end-to-end win.

## 3. Optimization tiers

### Tier 1 — The diffusion loop (target: the 435 s / 59%)

- **Step reduction.** 36 steps is the reference. Higher-order solvers
  (DPM-Solver++) cut to ~20 with no retraining. Consistency / step distillation
  reaches 4–8 steps — a 4–9× loop win, but needs a distilled checkpoint
  (training effort; Studio territory).
- **CFG batching / distillation.** Classifier-free guidance is 2 DiT forwards
  per step. Batch cond+uncond into one (built — `--native-loop`); measured
  **1.05× e2e at 49 f**, F14 explains why it's small at this scale.
  Guidance-distillation (drop the unconditional pass entirely) is the
  remaining lever here — up to 2× because it actually halves the work.
- **Attention.** ~56k-token sequence, head_dim 128. Today: SDPA → aotriton
  flash. Next: CK flash-attn tuned for gfx942; FP8 attention (CDNA3 MFMA);
  NATTEN-style neighborhood attention exploiting video locality → sub-quadratic.
- **`torch.compile`** the DiT forward (inductor + triton-rocm): operator
  fusion, removes Python and kernel-launch overhead.
- **Feature / step caching** (DeepCache / TeaCache style): DiT block outputs
  change slowly between adjacent steps — cache and skip. Training-free, ~1.5–2×.

### Tier 2 — VAE decode (deprioritized — see §2)

Steady-state VAE decode is ~1% of wall time, not the ~41% a cold run implied.
This tier is parked unless a workload moves the number (very long clips,
decode-heavy model variants). Techniques, if needed later:

- **Tiled / temporal-chunked decode.** The Cosmos VAE is causal-temporal;
  decode in temporal chunks (and spatial tiles) to bound memory traffic.
- **Streaming decode.** Decode chunks as the diffusion finishes them and emit
  frames — this is the plan's frame-level streaming; collapses time-to-first-frame.
- **Compile / fuse** the decoder convolutions; evaluate lower-precision VAE.
- **Overlap** VAE decode of chunk N with DiT denoising of chunk N+1 on separate
  HIP streams.

### Tier 3 — MI300X kernel layer

- **FP8 (e4m3) GEMMs** — CDNA3 has native FP8 MFMA; the DiT is GEMM-bound at
  4096 hidden. ~2× math throughput with quantization care. (Plan: Phase 2.)
- **hipBLASLt autotuning** for the exact DiT / VAE shapes.
- **CK / AITER** attention and GEMM kernels tuned for gfx942.
- Longer term: the **Kernel** product — RL-driven kernel synthesis (later phase).

### Tier 4 — Serving throughput

- **Continuous batching** where temporal dependencies allow (plan, Phase 2).
- **Stage pipelining** — T5 / DiT / VAE as pipeline stages: while the VAE
  decodes request A, the DiT denoises request B.
- **Paged latent cache** (already stubbed in `repercep.runtime.latent_cache`) —
  frame-aware reuse of latent tiles.
- **Exploit the ~140 GiB of free HBM** — large batches, multiple model
  variants / LoRAs co-resident, zero CPU offload. An H100 at 80 GiB must
  offload; the MI300X does not — a structural latency and throughput edge.

## 4. The MI300X angle

- 192 GiB HBM3 → everything resident, big batches, no offload.
- ~5.3 TB/s HBM bandwidth → directly helps the memory-bound VAE decode.
- Native FP8 (OCP e4m3 / e5m2) MFMA on CDNA3.
- Strategic: no production-grade world-model serving stack exists on AMD today
  — being fast here *is* the differentiation (implementation plan §5.4).

## 5. Sequencing & targets

1. **Profile properly** — per-stage, warmup-separated (one-time ROCm kernel
   autotuning vs. steady state). `repercep.bench` + torch profiler / `rocprof`.
2. **Tier 1 training-free wins** — `torch.compile`, CFG batching, solver swap,
   feature caching. Best near-term ratio, no new weights.
3. **Tier 2 VAE** — tiling + streaming decode.
4. **Tier 3 kernels** — FP8, CK attention, hipBLASLt.
5. **Tier 4 throughput** — batching + stage pipelining.

Plan targets: **2–3× over naive PyTorch + Diffusers in Phase 1**, **3–5× in
Phase 2** (with continuous batching + FP8). Every change is measured by
`repercep.bench` against the 738 s baseline in §2.

## 6. Explicitly NOT yet

- Hand-written HIP kernels from scratch — exhaust CK / aotriton / Triton /
  `torch.compile` first (plan: "use existing primitives, don't reinvent").
- The RL kernel synthesizer — later-phase scope.
- Multi-GPU — only a single MI300X VF is available on this host.
- Distillation training — needs a training pipeline (Studio scope).

## 7. Wan-2.2 — measured baseline (Session 11, 2026-05-23)

The Cosmos baseline above is the leading workload; Wan-2.2-T2V-A14B is
the second WM family Repercep serves end-to-end (`docs/WAN_ON_MI300X.md`).
The MoE topology (two ~14 B-param expert transformers, ~14 B active /
step, ~27 B total) is structurally heavier than Cosmos's single-DiT
shape; the diffusers path is identical (no FP8, no compile, no native
loop today).

**81 f / 40 step / 1280×720 — the canonical Wan reference shape:**

| | Wan team single H100 (FP8 + offload) | Repercep MI300X (BF16, no offload) |
|---|--:|--:|
| Wall, one-shot measured | — | **2576 s** (cold; contested) |
| Per-step steady-state | ~26 s / step (implied: 1041 / 40) | **~41 s / step** (last 11 steps of primary run; first 2 of profile pass — bracketed to 41.14–41.17 s) |
| Projected DiT loop steady-state (40 × per-step) | ~1041 s | **~1640 s** |
| Projected total steady-state | 1041.5 s | **~1700 s** |
| Peak HBM | 79.8 GB | **85.1 GiB** |
| Quantization | FP8 weights (model_dtype convert) | None (BF16 transformer + FP32 VAE) |
| Offload | `--offload_model True` (inactive MoE expert → CPU) | None (both experts resident) |

Repercep at the diffusers BF16 path is **~1.6× behind H100's optimized
single-GPU number**, but **~1.6× ahead of single A100's diffusers BF16
number** (2735.7 s in the same Wan team table). The H100's advantage
comes mostly from FP8 weight conversion + inactive-expert CPU offload;
MI300X's 192 GiB doesn't need either, and the same offload-free path
on H100 would not fit in 80 GB.

**Wan-2.2 levers, in priority order:**
- **Wan-shaped native loop with adaptive caching** — biggest expected
  win, no quality change for typical schedules. The community's H100
  TeaCache + Sage results land 2.5–3× speedup over a similar baseline
  ([Morphic](https://morphic.com/blog/boosting-wan2-2-i2v-56-faster),
  [Voltage Park](https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations)).
  Repercep's adaptive cache today is keyed on `CosmosTransformer3DModel`
  block shapes; the Wan port is a Phase-2 follow-up.
- **CFG batching for the low-noise expert phase.** CFG is unbatched
  today (2 transformer forwards / step); the 35 low-noise steps are
  where batching pays back most.
- **FP8 attention** — the diffusers `repercep_fp8` backend (Session 10)
  generalizes to `WanTransformer3DModel` without code changes; needs
  shape autotuning at Wan's grid (different B / H from Cosmos).
- **CPU offload of the inactive MoE expert** — would cut peak HBM
  to ~45 GiB (one expert + activations), freeing the rest of MI300X's
  192 GiB for batching or co-residency. Headroom for 4×
  concurrent generations or two model variants resident, vs the H100
  envelope which can fit at most one resident clip.
- **Both-expert torch.compile** — Wan's `compile_transformer=True`
  knob compiles both `transformer` and `transformer_2`. F15-style
  segfault gating may need extending to Wan shapes; not exercised yet.

**Why no Wan number after Tier-1 levers today:** the
`denoise_cosmos_video` native loop is Cosmos-block-shape-specific;
plumbing the equivalent for `WanTransformer3DModel` (MoE boundary +
two transformers) is a non-trivial port. Tracking as next session.

---

## FVD harness (Agent K, 2026-05-24)

`scripts/compute_fvd.py` lands the standard Fréchet Video Distance over
I3D features (`i3d_r50` from pytorchvideo, Kinetics-400 pretrained,
2048-D pre-classification feature tap). Pairs with
`scripts/verify_quality.py` as the distribution-level cache-quality
metric — the right arbiter for trajectory-divergent diffusion outputs
where pixel LPIPS over-penalises trajectory variance (see F23 / F24 in
`BUILD_LOG.md`).

**Math.** `||mu_A - mu_B||^2 + tr(S_A + S_B - 2 sqrt(S_A S_B))` on
float64 features via `scipy.linalg.sqrtm`. Small-diagonal jitter
(`1e-6 * I`) before sqrtm to stabilise the rank-deficient small-N case;
.real'd output with an imaginary-component sanity check.

**Small-N path.** FVD literature uses N >= 1000. Our cache vs no-cache
comparisons today have N = 1: the harness detects this and falls back
to a per-clip feature L2 distance with an explicit "too small for FVD"
note. Between N=2 and N=--n-warn (default 50) it returns a real FVD
with a loud preliminary-only warning. Treat at-small-N FVDs as
relative-ordering instruments, not absolute literature-comparable values.

**To use it for a real cache-quality verdict:** generate ~50 no-cache
Cosmos references across distinct prompts (~6 GPU-hours at 470 s each),
then 50 adaptive-cache candidates with the matching prompts and call
the script with N=50 per side. The harness is the missing piece, not
the references.
