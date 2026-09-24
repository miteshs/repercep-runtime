# Repercep walkthroughs — AVID (action-conditioned video diffusion): a **design** walkthrough

> ⚠️ **This is a design / spec walkthrough, not a code tour.** Unlike the
> [Cosmos](../walkthroughs/) and [V-JEPA 2-AC](../walkthroughs-vjepa2-ac/) series —
> which walk through *real, runnable* code — **AVID is not implemented in
> Repercep.** It is the deferred **Phase 3** of
> [ADR-0008](../adr/0008-interactive-world-model-seam.md). Its substance, an
> action-conditioning **adapter**, has to be *trained* (not just ported), so
> there's no code to tour. What follows is *how AVID would be built on the real
> machinery Repercep already has* — and it doubles as the Phase-3 implementation
> plan.

## What AVID is, and why it belongs here

**AVID** = adapt a **frozen, pretrained video-diffusion model** (Cosmos/Wan) into
an **action-conditioned world model** by training a small **adapter** — without
touching the base model's weights. It's the **pixel sibling of V-JEPA 2-AC**:

- V-JEPA 2-AC predicts the next *embedding* (latent, cheap, plans by energy).
- **AVID predicts the next *frame*** (pixels) by running a (cached) conditioned
  **denoise** step — so it produces watchable video while still being a
  step-with-an-action world model.

The reason it fits Repercep cleanly: it implements the **same
`InteractiveWorldModel` seam** as V-JEPA 2-AC, and it **reuses the Cosmos denoise
loop + adaptive cache** (Cosmos walkthrough Part 3). So the adaptive-cache
investment — which does *not* apply to V-JEPA's latent path — **does** pay off
here. One seam, two world-model families (latent + pixel); that's the structural
point of ADR-0008.

## How it's pitched

Same two axes as the other series: the **ML** (how an adapter conditions a frozen
diffusion model on actions, vs. V-JEPA's latent prediction and vs. full
fine-tuning) and the **systems** (how `step` becomes a cached conditioned denoise,
what's reused, what it costs). Every reference to *existing* Repercep code is a real
`file:line`; every AVID-specific piece is marked **(design)**.

## The path

| # | Part | What it covers |
|---|------|----------------|
| 0 | [Orientation](00-orientation.md) | what AVID is, why it fits the seam, the 10k-ft design + the honest real-vs-unbuilt split |
| 1 | [The method](01-the-method.md) | action-conditioned video diffusion via a frozen-base **adapter** — the ML, vs. V-JEPA (latent) and vs. full fine-tune |
| 2 | [Mapping onto Repercep](02-mapping-onto-repercep.md) | the `AvidEngine` design: `step` as a cached conditioned denoise reusing `denoise_cosmos_video` + the adaptive cache; populating `LatentStep.frame`; serving reuse |
| 3 | [Cost, training, build order](03-cost-training-build-order.md) | what's reused (cache/loop/seam/serving) vs. the real work (training the adapter); planning over pixels; the Phase-3 build order; open questions |

## What's real vs. design

- **Real (exists, runnable, reused):** the `InteractiveWorldModel` seam
  (`runtime/interactive.py`), the native denoise loop + adaptive cache
  (`runtime/denoise.py`), the `LatentStep.frame` pixel envelope (`runtime/
  types.py`), the `/v2/world/session` WebSocket (`serving/app.py`), and the
  attention-dispatch seam AVID would inject action conditioning through
  (Cosmos walkthrough Part 4).
- **Design / unbuilt:** the AVID **adapter** (needs training on action-labelled
  video + a GPU) and the `AvidEngine` class. These are marked **(design)**
  throughout and are the actual Phase-3 work.

> Attribution: AVID — "Action-conditioned Video Diffusion," adapting a frozen
> pretrained video-diffusion model to a world model via an adapter (RLC 2025;
> indexed in the diffusion-for-robotics literature).
>
> Status: **Parts 0–3 complete** — the full AVID design / ADR-0008 Phase-3 plan
> (a design walkthrough, not a code tour; AVID is not yet built).
