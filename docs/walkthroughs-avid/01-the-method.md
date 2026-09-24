# Part 1 — Action-conditioning a frozen diffusion model

The ML question AVID answers: you have a strong pretrained *text→video* diffusion
model (Cosmos), and you want a model that predicts **the next frames given an
action**. How do you get there cheaply, without wrecking the video prior you paid
so much for?

## 1.1 Three ways, and why AVID picks the third

1. **Fine-tune the whole DiT** on action-labelled video. Expensive (7 B params),
   needs a lot of in-domain data, and risks **catastrophic forgetting** of the
   general video prior. You'd be re-paying for the base model.
2. **Train a separate latent predictor** — that's the **V-JEPA 2-AC** path
   ([sibling series](../walkthroughs-vjepa2-ac/)): cheap, plans well, but predicts
   *embeddings*, not pixels. Different model, different strengths.
3. **Freeze the base, train a small adapter** — **AVID**. Keep the pretrained
   video DiT's weights fixed; train a small **action-conditioned** module on a
   *modest* set of action-labelled videos that **steers the frozen model's
   denoising** to follow the action. You keep the photorealism and the data bill
   is small.

AVID picks #3 because the goal is **pixel** rollouts that look like the base
model's video, with action control, on a budget — and because (per the AVID
result) you often don't even have access to the base model's parameters, only its
sampling. That's the same spirit as ControlNet / T2I-Adapter for image diffusion,
applied to a *video world model*.

## 1.2 How the adapter conditions denoising

A diffusion model, at each denoise step, predicts "which way is less noisy" for
the current latent (Cosmos walkthrough Parts 2–3: the DiT outputs a noise/clean
prediction the scheduler steps along). AVID makes that prediction
**action-aware**:

- The **frozen base** contributes its usual prediction (the strong, general video
  prior).
- A **small action-conditioned model (the adapter)** contributes a **correction**
  conditioned on the action (and recent frames).
- The two are **combined at inference** (AVID's recipe is a learned combination of
  the frozen base output and the adapter's action-conditioned output — think
  guidance-style mixing), so the denoise trajectory bends to follow the action
  while staying on the base model's manifold.

Train *only the adapter*, on action-labelled clips, to make the combined denoise
reproduce the true next frames. The base never moves. (This is the level the AVID
paper establishes; the exact adapter architecture and mixing weight are its
specifics — see the attribution in the [index](README.md).)

## 1.3 From "denoise a frame" to "world model"

One conditioned denoise gives you the next frame(s) given the current frames + an
action. Make it a **world model** by rolling it forward:

```
frame_0 (+ encode)  --action a1-->  conditioned denoise  -->  frame_1
frame_1             --action a2-->  conditioned denoise  -->  frame_2   ...
```

Each step is a (short) denoise of the next chunk, conditioned on the action and
the previously generated frames. That's a **closed-loop, action-conditioned,
pixel** world model — the thing Part 2 maps onto Repercep's seam.

## 1.4 Why pixels (the trade vs. V-JEPA)

- **For:** the rollout is *watchable and verifiable* — you can look at it, score
  it, hand it to a human, or use it as synthetic training data; and it inherits
  the base model's photorealism. V-JEPA's latent rollout can't be inspected
  directly.
- **Against:** each step is a **denoise** (many DiT forwards), not a single tiny
  predictor call. That's orders of magnitude more compute per step than V-JEPA's
  latent predict — which is precisely why **the adaptive cache (Cosmos Part 3)
  matters here** and why **planning over AVID by rolling out many candidates is
  expensive** (Part 3). Pixels buy inspectability; you pay in compute.

So AVID and V-JEPA 2-AC aren't competitors — they're two points on a trade-off,
both reachable through the *same* interactive seam. That's the design's whole
appeal.

## 1.5 The honest part

AVID is an **external method**; Repercep would *implement* it on top of Cosmos/Wan.
The load-bearing piece — the **trained adapter** — requires action-labelled video
for the target domain (robotics, driving, a game), a training run, and a GPU.
There is no shortcut and no stub that substitutes for it; everything in Part 2 is
the *plumbing* that the trained adapter would slot into.

**Next:** Part 2 — the `AvidEngine` design: `step` as a cached conditioned denoise
that reuses `denoise_cosmos_video`, and how the action conditioning hooks into the
attention-dispatch seam.
