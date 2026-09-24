# Part 3 — Cost, training, and the build order

The honest accounting: what AVID reuses (real, free), what it actually costs per
step, what the real ML work is (training the adapter), and the order I'd build it.
This part *is* the ADR-0008 Phase-3 plan.

## 3.1 Reused vs. the work

| Reused (real, exists today) | The actual work (design / Phase 3) |
|---|---|
| `InteractiveWorldModel` seam (`runtime/interactive.py`) | Train the **action adapter** (data + GPU) — the load-bearing piece |
| `denoise_cosmos_video` + adaptive cache (`runtime/denoise.py`) | Extend the loop: `init_latent` (img2img-style start) + `action_conditioning` |
| Cosmos VAE encode/decode, FP8 kernels (Cosmos Parts 5–6) | Wire the conditioning **hook** (side adapter beside the dispatch seam) |
| `LatentStep.frame` envelope + `/v2/world/session` (V-JEPA Part 5) | The `AvidEngine` class + a stub-adapter CPU test |

The plumbing is mostly there. **The adapter is the deliverable**, and no plumbing
substitutes for it.

## 3.2 The cost model (why the cache matters here)

Per AVID step = a **short, cached** denoise + a VAE decode:

- A one-chunk denoise at `cache_mode="adaptive"` runs only a handful of full DiT
  forwards (Cosmos Part 3: ~11 of 36 at thr=0.3 for a full clip; far fewer for a
  short chunk), plus the cached reuses — so the adaptive cache **directly** cuts
  AVID's per-step cost. (This is the one world-model family where the cache lever
  applies — V-JEPA can't use it.)
- The **VAE decode** is the per-step memory/compute spike (Cosmos Part 6).

Net: an AVID step is "a fraction of a Cosmos generation," per step. That's fine
for **offline rollout / synthetic data / human-in-the-loop**, and marginal for
**real-time** control — which is why a distilled/edge base or Wan-TI2V (smaller)
may be the better base than the 7 B Cosmos DiT for latency-bound use (§3.6).

## 3.3 Training the adapter (the crux)

This is the real ML, and it can't be faked:

- **Data:** action-labelled video for the target domain (robot trajectories,
  driving logs, game inputs) — AVID's selling point is that a *modest* set
  suffices because the base is frozen.
- **Objective:** train *only* the adapter so the combined (frozen base + adapter)
  denoise reproduces the true next frames given the action (Part 1 §1.2).
- **Compute:** a GPU and a training run — far less than pretraining the base, but
  real. The base never moves.

Everything in Part 2 is the harness this slots into.

## 3.4 Planning over pixels — don't (directly)

`plan()` over AVID is expensive (each candidate = a full denoise rollout, Part 2
§2.6). The recommended design is the **hybrid** the shared seam enables:

> Plan in **V-JEPA latent space** (cheap energy-MPC, V-JEPA Part 4) to choose
> actions; **render** the chosen actions with **AVID** for watchable, verifiable
> pixel rollouts.

Latent for *deciding*, pixels for *showing* — each path doing what it's cheap at.

## 3.5 The build order (Phase 3)

1. **Plumbing first, no adapter.** Add `init_latent` + an (initially no-op)
   `action_conditioning` to `denoise_cosmos_video`; write `AvidEngine.step` around
   it; CPU-test with a stub `pipe`/adapter (mirroring how `vjepa2_ac` is tested).
   With a no-op adapter it just re-denoises Cosmos — validates the harness end to
   end without weights.
2. **Train the adapter** on a small action-labelled set (GPU) — the real ML.
3. **Serve** through the existing `/v2/world/session`; `LatentStep.frame` carries
   the rendered frames; verify on a GPU box.
4. **(optional) Hybrid planning** — V-JEPA latent MPC choosing actions, AVID
   rendering them.

Steps 1, 3, 4 are mostly real-code/wiring on the existing machinery; **step 2 is
the irreducible ML investment.**

## 3.6 Open questions

- **Per-frame vs per-chunk denoise** — one frame per step (responsive, less
  temporally coherent) vs k frames (coherent, laggier). A latency↔coherence dial.
- **Cache × action responsiveness** — the adaptive cache reuses noise predictions
  across steps based on input similarity (Cosmos Part 3 §3.4). With the *action*
  changing each step, does the rel-L1 gate still skip safely, or does it blur the
  action's effect? An empirical finding waiting to happen — measure before
  trusting the cache on AVID.
- **Conditioning injection point** — side adapter (recommended) vs cross-attn/adaLN.
- **Which base** — Cosmos-7B (quality), Wan-TI2V-5B (lighter), or a distilled/edge
  model (latency). The frozen-base design makes swapping the base cheap.

## 3.7 Honest closing

AVID is the design that **unifies Repercep's diffusion stack** (the loop, the
adaptive cache, the FP8 kernels, the VAE) **with its interactive seam** — the one
world-model family that uses *all* the existing machinery, on the same
`InteractiveWorldModel` interface as V-JEPA 2-AC. It is genuinely **buildable on
what's already here**; what it is *not* is built — its value is a **trained
adapter**, which is real ML work on a GPU. This walkthrough is the plan, not the
product, and it's labelled that way throughout so nobody mistakes a design for a
shipped feature.

```
ResetRequest → reset: VAE-encode seed → WorldState (latent window)            §2.3 (design)
  → loop:  Action → step: cached conditioned denoise (frozen DiT + adapter)   §2.4 (design)
                          → VAE decode → next frame                            (reuses Cosmos Parts 3,5,6)
           streamed as LatentStep(frame=...) over /v2/world/session           (reuses V-JEPA Part 5)
  → plan: deferred → use V-JEPA latent planning + AVID rendering (hybrid)      §3.4
```

That's the AVID design — the pixel world model on the same seam, when someone's
ready to train the adapter.
