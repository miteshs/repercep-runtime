# Part 1 — JEPA and energy-based models

This is the ML-foundations part. By the end you should understand *why* V-JEPA
2-AC is shaped the way it is — why it predicts embeddings instead of pixels, what
"energy" actually means here, and why that makes planning cheap. It connects
directly to the code: the `energy` you'll see in Part 4 (`torch.linalg.vector_norm(
s_T − goal)`) is literally the quantity defined below.

## 1.1 Two ways to model the world

- **Generative** (diffusion, autoregressive-pixel): learn `p(x)` over *all the
  pixels*. To use it you must **render everything** — including the parts that are
  fundamentally unpredictable (exact leaf positions, fine texture, sensor noise).
  Cosmos is here (the whole [Cosmos walkthrough](../walkthroughs/) is a generative
  model).
- **Joint-Embedding Predictive (JEPA)**: don't model pixels. **Encode** an
  observation to an embedding, and predict the *embedding* of what comes next.
  You only ever commit to the **predictable, abstract** content. Yann LeCun's
  argument: spending model capacity rendering unpredictable detail is waste; for
  understanding and planning you want the representation, not the pixels.

V-JEPA 2-AC is firmly in the second camp. That single choice — predict
representations, not pixels — cascades into everything: no VAE decode, a cheap
scalar to plan against, and a closed loop you can actually run interactively.

## 1.2 Energy-based models, properly

An **energy-based model (EBM)** learns a scalar **energy** `E_θ(x)` (or `E_θ(x,
y)` for a conditional one). Low energy = "this configuration is plausible /
compatible"; high energy = "implausible." You *can* turn it into a probability —
`p(x) = e^{−E(x)} / Z`, the Boltzmann form — but the normalizer
`Z = ∫ e^{−E(x)} dx` (the **partition function**) is intractable in general.

The liberating fact: **most things you want don't need `Z`.**
- To *compare* candidates (is `y₁` or `y₂` a better continuation of `x`?) you only
  need relative energies — `Z` cancels.
- To *plan* (find the `y` that best continues `x`) you want `argmin_y E(x, y)` —
  `Z` is a constant w.r.t. `y` and drops out.

So an EBM that never computes `Z` is still fully useful for **comparison and
search** — which is exactly planning. The catch moves to *training* (how do you
learn `E` without `Z`?) and *inference* (how do you find the argmin?).

- **Inference** = minimize energy over candidates: gradient descent on `y`, or
  sampling (Langevin), or **search over a set of candidates** (the route Repercep
  takes — CEM, Part 4).
- **Training without `Z`**: contrastive (push energy down on real pairs, up on
  fake — needs negative sampling), score matching, or — JEPA's route —
  **non-contrastive regularization** (next section).

## 1.3 JEPA *is* an energy-based model

Concretely, JEPA defines:

```
s_x  = Encoder(x)                  # embedding of the context
s_y  = TargetEncoder(y)            # embedding of the thing to predict (the "answer")
ŝ_y  = Predictor(s_x, z)           # predicted embedding (z = optional latent for ambiguity)
E(x, y) = ‖ ŝ_y − s_y ‖            # energy = prediction error IN EMBEDDING SPACE
```

Training pushes `E` down for real `(x, y)` pairs. But there's a trap: if the
encoder maps *everything* to the same constant vector, then `ŝ_y = s_y` trivially
and `E = 0` everywhere — **representation collapse**. The whole game in JEPA is
preventing that *without* the partition function:

- **VICReg**-style regularizers: explicitly keep the embedding's variance up and
  decorrelate its dimensions, so a constant solution is penalized.
- **EMA target encoder + stop-gradient** (the I-JEPA / V-JEPA route): the
  `TargetEncoder` is an exponential-moving-average copy of the online encoder, and
  gradients don't flow through it. The moving target can't be gamed into a
  constant, so the predictor must actually predict. This is a distillation-style
  trick — and it's *why JEPA sidesteps `Z`*: it never normalizes a distribution;
  it just regularizes against collapse.

That's the ML heart: **a learned compatibility energy in embedding space, trained
non-contrastively.** No softmax over pixels, no `Z`.

## 1.4 V-JEPA → V-JEPA 2 → V-JEPA 2-AC

- **I-JEPA** (images, CVPR 2023): mask image regions, predict their embeddings
  from the visible context, EMA target encoder.
- **V-JEPA / V-JEPA 2** (video, Meta 2025; arXiv 2506.09985): the same recipe over
  spatiotemporal video — mask tubes, predict their embeddings. V-JEPA 2 scales the
  encoder (the ViT-g you'll meet in Part 2) and shows the embeddings transfer to
  understanding, prediction, and planning.
- **V-JEPA 2-AC** (action-conditioned): post-train a small (~300 M) **block-causal**
  predictor head on a modest amount of robot interaction data. It **autoregressively
  predicts the next state embedding conditioned on an action and the previous
  states**. That turns the encoder-predictor into a **world model for control**.

## 1.5 Planning = energy minimization (the bridge to Part 4)

Here's where it pays off. Give the model a **goal embedding** `g`. Define the
energy of a candidate action sequence `a₁..a_H` as the distance between where the
*rolled-out* world model thinks it lands and the goal:

```
E(a₁..a_H) = ‖ rollout(s₀, a₁..a_H) − g ‖
```

Planning is `argmin_{a} E(a)` — model-predictive control as **energy
minimization**. Repercep implements it with the **cross-entropy method** (CEM):
sample action sequences, roll each out, keep the lowest-energy ("elite") ones,
refit, repeat. That's exactly `_rollout_energy` / `_plan_sequence` / `plan` in
`models/vjepa2_ac.py` (Part 4) — and the `1.763 → 0.401` you saw in Part 0 is this
energy dropping as CEM converges.

## 1.6 Why this is cheap (the systems payoff)

Energy here is a **scalar distance in embedding space** — not a normalized
probability, not a rendered video. So evaluating one imagined future is: a few
predictor forwards (small, block-causal) + one vector norm. You can evaluate
*dozens* of candidate action sequences per planning step and still be fast,
because you're comparing **embeddings, not pixels**. Contrast: planning by rolling
out a *diffusion* video model per candidate action would mean a full multi-second
generation per candidate — absurd. **Predicting in representation space is what
makes closed-loop, search-based planning tractable at all.** That's the deep
reason this path, not the diffusion path, is the one suited to interactive control.

## 1.7 Honest framing

This is LeCun's bet, not a settled result. Whether energy-based / JEPA world
models beat autoregressive (Genie, Oasis) or diffusion world models *empirically*
at scale is open. What's not in dispute: V-JEPA
2-AC is an **open** on-ramp to the regime, it's an energy-based model by
construction, and the energy-minimization-at-inference idea is generalizing fast
(e.g. **Energy-Based Transformers**, arXiv 2507.02092, frame *any* prediction as
inference-time energy descent — "System 2 thinking").

## Run it (CPU)

The energy is just a distance — here it is in isolation (this is exactly what
`_rollout_energy` returns at its last line):

```python
import torch
predicted_state = torch.tensor([0.2, -0.1, 0.4, 0.0])
goal            = torch.tensor([0.0,  0.0, 0.0, 0.0])
energy = torch.linalg.vector_norm(predicted_state - goal)   # lower = closer to the goal
print(float(energy))   # 0.458...
```

The full planner that *minimizes* this over action sequences runs via
`scripts/run_vjepa2_ac.py --stub --plan` (Part 0 / Part 4).

**Next:** Part 2 — the actual model: the V-JEPA 2 encoder (HuggingFace ViT) and
the AC predictor, and what's real vs. scaffold in `models/vjepa2_ac.py`.
