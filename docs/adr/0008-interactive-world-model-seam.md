# ADR-0008 — Interactive world-model seam (action-conditioned, closed-loop)

- **Status:** Accepted (seam); implementation phased (see below)
- **Date:** 2026-05-28
- **Relates to:** ADR-0001 (MI300X-first), ADR-0003 (vendor-neutral Backend
  Protocol)

## Context

Repercep's engine seam today is `repercep.runtime.engine.WorldModelEngine`:
`generate(request) -> Iterator[Frame]` — one request in, one stream of RGB
frames out. It models *request→response batch generation*, which is exactly
what the general-purpose video-diffusion servers (xDiT, FastVideo, vLLM-Omni,
SGLang-Diffusion) also do, and increasingly do well.

The defensible regime is the one those
servers structurally do **not** model: a *stateful, action-conditioned, closed
loop* — the client injects an action each step, a persistent world-state latent
is advanced and reused, and the metric is closed-loop latency under state
carryover, not batch wall-time. That is also the energy-based / JEPA world-model
regime (Yann LeCun's program): Repercep already runs a member of it, V-JEPA 2, but
only as a benchmark script (`scripts/run_vjepa2.py`), not as an engine.

Two facts force a new seam rather than an extension of `generate`:

1. The loop is **bidirectional and stateful** — there is no point up front at
   which the full action sequence is known, so a single `generate(request)`
   cannot express it.
2. A latent world model (V-JEPA 2-AC) predicts the **next state embedding**, not
   pixels. Its step output is a latent + a scalar energy + (optionally) a
   planned action — the `Frame`/RGB contract does not apply.

## Decision

Add a second engine Protocol,
`repercep.runtime.interactive.InteractiveWorldModel` (`reset → step(action) → … /
plan`), alongside — not replacing — `WorldModelEngine`:

- **Lead implementation:** `repercep.models.vjepa2_ac.VJepa2ACEngine` — V-JEPA 2-AC
  (~300M, block-causal), a latent energy-based world model. `plan()` is
  energy-minimizing MPC: sample candidate action sequences, roll out, score each
  by the embedding-space distance to a goal (the energy), return the best first
  action.
- **Pixel sibling (later):** an AVID-style action-conditioned video-diffusion
  engine implements the *same* Protocol but decodes pixels per step, reusing the
  native denoise loop and the adaptive cache.
- The one-shot `WorldModelEngine.generate` becomes expressible as a thin facade
  over the interactive seam (reset + N auto-stepped frames).

New wire types live in `repercep.runtime.types`: `Action`, `RolloutParams`,
`ResetRequest`, `WorldState` (in-process, tensor-carrying, like `Frame`), and
`LatentStep` (the streamable envelope, like `FrameChunk`).

## Rationale

1. **Structural differentiation.** Request/response diffusion servers will keep
   absorbing batch tricks (caching, sequence parallelism); a stateful interactive
   loop is a different shape they don't target. This is the
   differentiation the project targets.
2. **Demand-side fit.** Maps to robotics/AV synthetic-data-in-the-loop and to
   NVIDIA Cosmos-Predict2.5 *video2world* action conditioning — where world-model
   inference actually gets deployed.
3. **Reuses the ADR-0003 seam.** The engine is one more class on the unchanged
   `Backend` Protocol; nothing below the seam changes. V-JEPA 2 already runs on
   the Backend (`scripts/run_vjepa2.py`), so the encoder path is proven.
4. **It is the energy-based world model, served.** Answers "what would an EBM in
   this stack look like" concretely: JEPA-as-EBM with energy-MPC planning.

## Consequences

- Latent world models produce **embeddings + energy, not pixels** — hence
  `LatentStep` (not `Frame`) and `decode_pixels=False` by default.
- The **adaptive denoise cache does not apply** to V-JEPA 2-AC (no denoise loop).
  It *does* apply to the AVID pixel sibling — that path keeps the cache
  investment intact.
- Serving needs a **bidirectional channel** (WebSocket session) the one-shot
  NDJSON stream can't provide — deferred to Phase 2.
- The AC predictor head + planner currently live in `facebookresearch/vjepa2`,
  not as an HF `AutoModel` — a small **port**, not a `from_pretrained`. The
  model-dependent engine methods land as `NotImplementedError` scaffolds with
  the intended bodies specified in their docstrings.

## Phased plan

0. Promote `scripts/run_vjepa2.py`'s encoder into a real engine (encode →
   embeddings) with CPU tests. *(proven path)*
1. Wire the `InteractiveWorldModel` seam + the latent rollout + the `plan()`
   energy-MPC, unit-tested against a stub predictor. **Done** — the
   model-agnostic loop and CEM/energy planner are implemented and green on CPU
   (`VJepa2ACEngine` takes an injectable encoder + predictor). Remaining: the
   encoder + AC-predictor *weight loading* and image/video URI decode — they
   need the real checkpoints + a GPU and raise `NotImplementedError` with the
   intended body in their docstrings.
2. Bidirectional WebSocket serving session (`/v2/world/session`) over the
   existing `create_app`; engine calls run in a threadpool (same rationale as
   the v2 driver thread). **Done** — reset/step loop tested end-to-end via the
   FastAPI TestClient against a stub engine.
3. AVID adapter on Cosmos/Wan for the pixel path that reuses the cache.

Phases 0–2 are CPU-runnable; the V-JEPA 2 ViT-L encoder is 0.3B.

## Revisit if

The deployable world-model demand turns out to be dominated by autoregressive
real-time models (Genie/Oasis-style) rather than diffusion/JEPA — in which case
the seam stays but the lead implementation changes.
