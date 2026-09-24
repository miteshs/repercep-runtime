# VLA (OpenVLA-7B) on H100 — Phase 1+2 verified (2026-07-26)

**One line:** the `VLAEngine` seam runs real **openvla/openvla-7b** end-to-end on
an H100 with **exact** action parity (0.0), and the candidate-batching lever
measures **5.4×–8.8×** — the leaderboard row the strategy names, on real weights.

## Setup

- **Box:** RunPod H100 80GB HBM3, `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- **Env:** torch 2.4.1+cu124 (image), `transformers==4.40.1`, `tokenizers==0.19.1`,
  `timm==0.9.10` (`--no-deps`), `accelerate`. OpenVLA loads via `trust_remote_code`,
  bf16, `attn_implementation="sdpa"` (the ROCm-portable path; no flash-attn build).
- **Model:** `openvla/openvla-7b` (MIT, ungated), ~14 GB, `unnorm_key="bridge_orig"`.
- **Repro:** `python -O scripts/run_vla_openvla_gpu.py --candidates 8,16,32`
  (self-contained: the OpenVLA `_VLAPipeline` + parity gates + bench).

## Phase 1 — correctness (the gates)

The engine wraps real weights through `reset → step → plan`, with two parity gates:

| Gate | Result | Meaning |
|---|---|---|
| **greedy parity** (our detokenize vs OpenVLA's own `predict_action`) | **max err 0.0e+00** | our replicated `vocab_size - id → bin_center → unnormalize(q01/q99)` is exact |
| **batch parity** (8 identical inputs, greedy → each vs batch-1) | **max err 0.0e+00** | the batched candidate path is numerically identical to batch-1 |

`step_index` advances; `plan` returns a 7-DoF action `[0.0009, -0.0063, -0.0023,
0.0018, -0.0215, 0.0161, 0.9961]` (gripper≈1.0). Peak HBM 23.9 GiB.

## The batch>1 finding (the interesting bit)

OpenVLA's stock remote code refuses batched generation in **two** places, both
*artificial* — the fusion math is batch-safe (`vision_backbone` / `projector` /
the `cat` on `dim=1` all act on the batch dim):

1. `prepare_inputs_for_generation` **raises** `"Generation with batch size > 1 is
   not currently supported!"` — comment: *"simplified for batch size = 1"*. We
   lift it with a one-method monkeypatch (`_enable_batched_generation`).
2. `forward()` **asserts** `input_ids.shape[0] == 1` in the cached-decode branch.
   We strip it by running under `python -O`.

Neither changes the computation — proven by the **batch-parity gate above being
exactly 0.0**. This is the honest way to unlock the lever: lift the guard, then
*prove* outputs are unchanged, rather than trust the workaround.

## Phase 2 — the candidate-batching lever

`plan()` decodes N action-token candidates and scores them. Baseline: a
per-candidate loop (N decodes at batch=1). Optimized: one batched decode
(N sequences sharing the prompt+vision prefix). Both sampled (`temperature=1.0`),
median of 12, warm.

| N candidates | per-candidate loop | batched | **speedup** |
|---:|---:|---:|---:|
| 8  | 985 ms | 182 ms | **5.43×** |
| 16 | 1963 ms | 264 ms | **7.43×** |
| 32 | 3905 ms | 443 ms | **8.83×** |

**Why so much larger than V-JEPA-AC's 1.6×** (`bench_cem_batched.py`): that model's
per-candidate forward runs over a ~2048-token context and is already
compute-bound, so batching only recovers GPU efficiency. OpenVLA's per-candidate
decode is **7 action tokens** — batch=1 massively underuses the H100, so batching
the shared prefix is a much bigger win, and it **grows with N** (5.4× → 8.8×).
This is exactly the "planning-decisions/sec under a latency budget" metric the
revised strategy stakes out, and nothing (vLLM, NIM) optimizes it for VLAs.

## Honest caveats

- **The scorer is still the open question, not the decode.** The lever is a
  *throughput* win on decoding N candidates; the `score_candidates` used here is
  a consensus placeholder (nearest the candidate mean). A real policy scorer
  (value / short rollout / reward model) is the modeling work Phase-1 does not
  settle — the serving-lever result stands regardless.
- **OpenVLA is stateless obs→action**, so `step` keeps the observation context;
  a real robot loop re-encodes each new frame (the seam's v0 doesn't thread new
  observations through `step` — noted in `VLA_PORT_PLAN.md`).
- **Vendor-neutral: MI300X verified too** (`docs/VLA_ON_MI300X.md`, same day) —
  identical parity 0.0 and the same 5–10× lever on AMD, no code changes.
  `attn="sdpa"` kept that path open.
- **Not committed as a default engine.** The verified pipeline lives in
  `scripts/run_vla_openvla_gpu.py` (injected, reproducible); promoting it to a
  typed `models/vla_openvla.py` wired into `build_pipeline` is a follow-up.
