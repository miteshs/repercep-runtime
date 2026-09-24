# Part 3 — The denoise loop

Part 2 was one DiT forward (noisy latent → noise prediction). This is the loop
that calls it 36× and turns pure noise into a clean latent. It's **Repercep's own
code** — `src/repercep/runtime/denoise.py` (`denoise_cosmos_video`), a from-scratch
re-implementation of diffusers' `CosmosTextToWorldPipeline.__call__` (Part 1)
with two additions: **CFG batching** and the **adaptive cache**. `CosmosEngine`
calls it when `use_native_loop=True` (`cosmos.py:261`).

## 3.1 Skeleton

`denoise_cosmos_video` (denoise.py:64) is a thin wrapper whose only job is the
`inference_mode` gate (130); `_denoise_impl` (153) is the real loop:

1. `encode_prompt` (180) → cond + uncond text embeddings. **T5 runs here, once.**
2. `set_timesteps` (193) → the 36 sigmas.
3. `prepare_latents` (197) → `randn × sigma_max` (pure noise).
4. CFG pre-batch (216) → `encoder_pair = cat([neg, pos])`.
5. the loop (233) — §3.3.
6. un-normalize + `vae.decode` + postprocess (352-371) → the video tensor (Part 6).

## 3.2 The systems lesson on line 130 (finding F18)

The comment at 123-129 is one of the most instructive in the repo:

> diffusers' `__call__` is `@torch.no_grad()`-wrapped; without the same gate here
> every step's autograd graph stays alive across the loop and activation memory
> grows ~linearly in step count. 17f/8 fits (~28 GiB); 121f/36 OOMs at ~189 GiB
> on a 192 GiB MI300X.

ML↔systems bridge: inference creates no gradients, but PyTorch doesn't *know*
that unless told. `torch.inference_mode()` (a strict superset of `no_grad`) tells
it — turning a 189 GiB OOM into a 52.5 GiB run. The bug hid in smoke configs
(small step counts) and only bit at the full reference shape: the canonical
"works at 17f, OOMs at 121f" trap.

## 3.3 The loop — CFG, the two-call scheduler, the cache

Per step (233-350), three things happen.

### (a) Prep the model input
```python
latent_model_input = pipe.scheduler.scale_model_input(latents, t).to(dtype)   # 234
timestep = t.expand(latents.shape[0]).to(dtype)                               # 235
```
`scale_model_input` is EDM's input preconditioning (scale by ~1/√(σ²+σ_data²)).

### (b) CFG batching — one batch-2 forward, not two (290-300)
```python
batched_input    = latent_model_input.repeat(2, 1, 1, 1, 1)   # [uncond ; cond]
batched_timestep = timestep.repeat(2)
noise_pred = pipe.transformer(hidden_states=batched_input,
                              encoder_hidden_states=encoder_pair, ...)[0]
```
The diffusers reference runs **two sequential** batch-1 forwards (Part 1, lines
575 & 586). Repercep folds them into **one batch-2** forward (position 0 = uncond,
1 = cond — matching the reference). The padding mask is deliberately *not*
pre-batched: the transformer repeats it internally by batch size (note 210-213).

Honest result (**F14**): ~1.05× end-to-end — essentially a **wash**. CFG batching
cuts *Python/launch overhead*, but the DiT is GEMM-bound, so halving launches
barely moves wall time. The durable lesson the project drew: *work-reducing
levers scale; overhead-cutting levers don't, at this size.* The native loop earns
its keep not here, but as **the seam where the cache lives** (§3.4).

### (c) The two-call EDM scheduler + the rewind (333-350)
```python
sample = torch.cat([latents, latents], dim=0)
noise_pred = pipe.scheduler.step(noise_pred, t, sample, return_dict=False)[1]   # -> x0
pipe.scheduler._step_index -= 1                                                 # rewind!
noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)  # CFG on x0
latents = pipe.scheduler.step(noise_pred, t, latents, pred_original_sample=noise_pred)[0]
```
Subtle, and an exact mirror of diffusers (Part 1, 598-608): the scheduler is
called **twice per step** with a manual `_step_index -= 1` between them. The first
call converts the raw noise prediction to a clean-sample estimate **x0**; CFG is
applied **on x0**; the second call uses that to actually advance the latent. The
rewind undoes the step-index increment from the first (diagnostic) call so the
second is the real one. Get the rewind wrong and the sigma schedule desyncs.

ML note — **CFG**: `cond + w·(cond − uncond)` with `w = guidance_scale = 7.0`
extrapolates *away* from the unconditional prediction, sharpening prompt
adherence. The unconditional branch uses Cosmos's long built-in negative prompt.

## 3.4 The adaptive cache — Repercep's real lever (237-282)

This is the 2.75× in the H100 headline. The idea: **late in sampling, consecutive
steps' inputs barely change, so the DiT's output barely changes — reuse it and
skip the forward entirely.** One skipped step = one whole batch-2 DiT forward not
run (28 blocks × S ≈ 109k bidirectional attention). That's why it dwarfs the
overhead-cutting levers.

The state machine, per step:
```python
if in_warmup or is_last or cached_noise_pred is None:   # quality floors -> always full
    should_skip = False
elif steps_since_full >= cache_force_full_every:        # periodic floor -> full
    should_skip = False
else:                                                    # the gate:
    diff   = (latent_model_input - last_full_input).abs().mean()
    denom  = last_full_input.abs().mean().clamp_min(eps)
    rel_l1 = diff / denom                                # relative L1 change of the input
    accumulated_rel_l1 += rel_l1
    should_skip = accumulated_rel_l1 < cache_adaptive_threshold
```
On a real forward it caches `noise_pred`, resets the accumulator, and updates
`last_full_input` (319-325); on a skip it reuses `cached_noise_pred` (285).

**This is TeaCache, and the code says so** (docstring 22-32). The honest caveat:
TeaCache's clever part is an *offline polynomial rescaler* that predicts output
drift from input drift; Repercep's v0 uses an **identity rescaler** (raw `rel_l1`
accumulation) — "a sound conservative default" the paper falls back to without a
rescaler. So: a competent re-implementation, not novel IP, and not a differentiator
on its own.

Quality is a **dial, not free**: lower `threshold` → fewer skips → closer to
no-cache but slower. At `threshold=0.3` the gate runs ~11 full forwards instead
of 36; the output is *trajectory-divergent* (LPIPS ~0.64 vs no-cache) but a valid
Cosmos generation with ~28-34 % less inter-frame motion — see `docs/METHODOLOGY.md`
§4 (the F23 retraction of the earlier "quality-preserved" claim). `DenoiseStats`
(45) records the full/skip counts `scripts/bench_caching.py` reports.

## Run it (CPU, no weights)

The cache state machine is exercised on CPU against a torch-only fake pipe in
`tests/test_denoise.py` — it injects a per-step-varying input so the gate has
something to measure, then asserts the full-vs-skip counts per mode:

```bash
python -m pytest tests/test_denoise.py -q     # in the [dev] env
```

**Next:** Part 4 — follow `dispatch_attention_fn` (Part 2 §2.5) down through
diffusers' backend registry to either SDPA→aotriton or Repercep's FP8 bridge.
