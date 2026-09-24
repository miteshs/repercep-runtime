# Repercep walkthroughs — understanding inference, line by line

A guided tour of how a generation request becomes video frames, from the
outermost API call all the way down to the GPU kernel.

> **Companion series:** for the interactive, energy-based path (V-JEPA 2-AC) —
> the closed-loop world model, not the one-shot generator — see
> [`../walkthroughs-vjepa2-ac/`](../walkthroughs-vjepa2-ac/).

**Who this is for / how it's pitched.** Someone who wants *every detail* on
**both axes**:

- the **ML / model level** — what the diffusion sampling and the transformer
  architecture actually do, including the mechanics and the math that matter
  (the score/noise objective, attention, adaLN conditioning, CFG); and
- the **systems / infrastructure level** — how it executes: memory layout,
  dtypes, the dispatch chain, the kernel, and how it lowers to the ISA.

We build up fundamentals where needed (no background assumed on either side) but
**don't stop at intuition** — the later parts do the real mechanics on both
sides. Every claim is grounded in a real `file:line` you can open; runnable
snippets are marked.

## The path — Cosmos-Predict-7B inference, outermost → kernel

| # | Part | What you'll understand |
|---|------|------------------------|
| 0 | [Orientation](00-orientation.md) | The whole request→frames skeleton, one diagram, the fundamentals in a page |
| 1 | [Model + weights](01-model-and-weights.md) | `from_pretrained`, the HuggingFace cache, safetensors sharding, what's *inside* the pipeline (T5 / DiT / VAE / scheduler) |
| 2 | [Architecture, layer by layer](02-architecture.md) | The Cosmos DiT (`CosmosTransformer3DModel`): patch embed → adaLN/timestep → blocks (self-attn, cross-attn, FF) → unpatchify, with exact tensor shapes |
| 3 | [The denoise loop](03-denoise-loop.md) | `runtime/denoise.py` line by line — timesteps, latents, the CFG two-call + rewind, the adaptive-cache state machine, VAE decode |
| 4 | [Dispatch + lowering](04-dispatch-and-lowering.md) | How a block's attention call goes `processor → diffusers dispatch → SDPA→aotriton` (default) **or** the Repercep FP8 bridge → Triton |
| 5 | [The kernel](05-the-kernel.md) | The FP8 Triton flash-attention kernel line by line (tiling, online softmax, FP8 `tl.dot`, autotune), and how Triton lowers to MFMA |
| 6 | [Execution + output](06-execution-and-output.md) | What runs per step, HBM/timing, VAE → pixels → mp4; the final tensor shape/dtype; the whole path end-to-end |

Read in order. Each part is self-contained enough to revisit.

## How to follow along

- **Reading the code:** every claim cites `path:line` — open them as you go.
- **Running things (no GPU/ROCm needed):** the runnable pieces work on CPU — the
  denoise control flow against a fake pipe, the kernel's Python plumbing, the
  `--stub` interactive engine. Each part has a **Run it** box where applicable.
- **The parts that need real hardware** (a live 121-frame generation, kernel
  timing) are marked **"run on your MI300X/H100"** and reasoned through precisely
  from the code instead.

## Honest scope

Repercep *wraps* HuggingFace `diffusers` for the Cosmos model itself. So the
"model" and "layers" (Parts 1–2) are largely diffusers' `CosmosTransformer3DModel`
and `AutoencoderKLCosmos`, while the denoise loop, the attention
dispatch/kernels, the backend seam, and the serving layer (Parts 3–6) are
Repercep's own code. The tour spans both and always says which is which.

> Status: **Parts 0–6 complete** — the full Cosmos inference path, outermost API
> → DiT layers → denoise loop → attention dispatch → gfx942 kernel → streamed
> frame, grounded in the real source + verified CPU runs. See
> `docs/SESSION_24_HANDOFF.md` for session context.
