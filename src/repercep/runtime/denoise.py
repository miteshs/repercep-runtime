"""Repercep-native Cosmos denoising loop.

Replaces the diffusers ``CosmosTextToWorldPipeline.__call__`` with a loop that
uses the same components (T5 encoder, DiT transformer, VAE, EDM-Euler
scheduler) but folds the classifier-free-guidance forward passes into a single
batched call. The diffusers loop runs two sequential transformer forwards per
step (BUILD_LOG F9); this collapses them into one batch-2 forward.

The data path mirrors the reference loop step-for-step — encoder pre-batching,
``scheduler.scale_model_input``, the scheduler's two-call pattern around CFG,
and the VAE postprocess — so behaviour stays bit-equivalent except for the
batched-vs-sequential difference in the transformer call.

Caching modes
-------------

* ``cache_mode="none"`` (default): every step runs a full DiT forward.
* ``cache_mode="fixed"``: legacy step-skip — after ``cache_warmup_steps``,
  run a full DiT forward only every ``cache_skip_every`` steps and reuse the
  cached ``noise_pred`` on the rest. The BUILD_LOG F16/F17 "rule of thumb"
  implementation.
* ``cache_mode="adaptive"``: TeaCache-style input-similarity gate. Maintains
  the *accumulated* relative L1 distance of the timestep-conditioned latent
  input vs. the last full forward; a step is skipped while that accumulator
  stays below ``cache_adaptive_threshold``. The last step, the warmup window,
  and every ``cache_force_full_every`` step always force a full forward as a
  quality floor. Reference: TeaCache (https://github.com/LiewFeng/TeaCache).

  The TeaCache paper fits an offline polynomial rescaler that predicts
  cumulative *output* drift from cumulative input drift; v0 uses an identity
  rescaler (raw ``rel_l1`` accumulation), which is what the paper falls back
  to when no rescaler is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_CACHE_MODES = ("none", "fixed", "adaptive")


class DenoiseStats:
    """Lightweight accumulator for adaptive-cache diagnostics.

    Populated by :func:`denoise_cosmos_video` when ``stats`` is passed. Used by
    ``scripts/bench_caching.py`` to report full-forward counts; not part of the
    runtime hot path.
    """

    __slots__ = ("full_forwards", "rel_l1_history", "skip_reasons", "skipped")

    def __init__(self) -> None:
        self.full_forwards: int = 0
        self.skipped: int = 0
        self.rel_l1_history: list[float] = []
        # One of "warmup", "force_floor", "accumulator_over_threshold",
        # "last_step", "no_cache_yet", "skipped" — one entry per step.
        self.skip_reasons: list[str] = []


def denoise_cosmos_video(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    height: int = 704,
    width: int = 1280,
    num_frames: int = 121,
    num_inference_steps: int = 36,
    guidance_scale: float = 7.0,
    fps: int = 30,
    seed: int | None = None,
    output_type: str = "pt",
    cfg_batched: bool = True,
    cache_skip_every: int = 0,
    cache_warmup_steps: int = 4,
    cache_mode: str = "none",
    cache_adaptive_threshold: float = 0.1,
    cache_force_full_every: int = 8,
    stats: DenoiseStats | None = None,
) -> Any:
    """Run the Cosmos denoising loop end-to-end and return a video tensor.

    Args:
        pipe: a loaded ``CosmosTextToWorldPipeline``.
        cfg_batched: if ``True`` (default), the per-step conditional and
            unconditional transformer forwards are folded into one batch-2
            call. If ``False``, the function uses the diffusers reference
            behaviour (two sequential forwards) as a correctness reference.
        cache_skip_every: if ``>= 2`` and ``cache_mode`` resolves to ``"fixed"``,
            after the warmup window run a full DiT forward only on every Nth
            step and reuse the cached output on the remaining N-1 steps.
            ``0`` (default) disables fixed-cadence caching.
        cache_warmup_steps: number of leading steps that always run a full
            forward, before caching kicks in. Defaults to 4.
        cache_mode: ``"none" | "fixed" | "adaptive"``. ``"none"`` disables all
            caching; ``"fixed"`` is the legacy ``cache_skip_every`` cadence;
            ``"adaptive"`` selects TeaCache-style input-similarity gating. For
            backward compatibility, if ``cache_mode == "none"`` and
            ``cache_skip_every >= 2``, the loop behaves as ``"fixed"``.
        cache_adaptive_threshold: gate threshold on the accumulated relative L1
            distance of the timestep-conditioned latent input (adaptive mode).
            Lower is more conservative.
        cache_force_full_every: in adaptive mode, force a full forward every N
            steps regardless of the gate. ``0`` disables the floor.
        stats: optional :class:`DenoiseStats` to populate with full/skip counts
            and per-step rel-L1 history. Used by ``scripts/bench_caching.py``.

    Returns:
        With ``output_type='pt'`` (default): a tensor shaped ``(B, T, C, H, W)``
        in ``[0, 1]`` — what ``pipe(...).frames`` returns for the same inputs.
    """
    import torch

    if cache_mode not in _CACHE_MODES:
        raise ValueError(
            f"cache_mode must be one of {_CACHE_MODES}; got {cache_mode!r}"
        )

    # Diffusers' own `CosmosTextToWorldPipeline.__call__` is wrapped with
    # `@torch.no_grad()`; without the same gate here every step's autograd
    # graph stays alive across the loop and activation memory grows
    # ~linearly in step count. 17f / 8 steps fits (~28 GiB peak); 121f /
    # 36 steps OOMs at ~189 GiB on a 192 GiB MI300X. `inference_mode` is a
    # strict superset of `no_grad` and the right gate since nothing here
    # is going to be backpropped through.
    with torch.inference_mode():
        return _denoise_impl(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
            seed=seed,
            output_type=output_type,
            cfg_batched=cfg_batched,
            cache_skip_every=cache_skip_every,
            cache_warmup_steps=cache_warmup_steps,
            cache_mode=cache_mode,
            cache_adaptive_threshold=cache_adaptive_threshold,
            cache_force_full_every=cache_force_full_every,
            stats=stats,
        )


def _denoise_impl(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str | None,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    fps: int,
    seed: int | None,
    output_type: str,
    cfg_batched: bool,
    cache_skip_every: int,
    cache_warmup_steps: int,
    cache_mode: str,
    cache_adaptive_threshold: float,
    cache_force_full_every: int,
    stats: DenoiseStats | None,
) -> Any:
    import torch

    device = pipe._execution_device
    transformer_dtype = pipe.transformer.dtype

    # 1. Encode prompts (cond + uncond).
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
        dtype=transformer_dtype,
    )

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # 2. Scheduler timesteps.
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 3. Initial latents.
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.config.in_channels,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=None,
    )
    padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

    # 4. Pre-batch encoder hidden states for CFG.  Order matches the diffusers
    # reference: position 0 = uncond, position 1 = cond.  The padding mask is
    # NOT pre-batched: the transformer repeats it internally by `batch_size`,
    # so a `(2, 1, H, W)` mask would double again to `(4, ...)`.
    encoder_pair: Any = None
    if cfg_batched:
        encoder_pair = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # Resolve effective cache mode for backward compatibility: callers that
    # still pass `cache_skip_every >= 2` with the default `cache_mode="none"`
    # get the legacy fixed cadence.
    effective_mode = cache_mode
    if effective_mode == "none" and cache_skip_every >= 2:
        effective_mode = "fixed"

    # 5. Denoising loop, optionally with step-skip caching after a warmup window.
    cached_noise_pred: Any = None
    last_full_input: Any = None  # Adaptive cache: input at last full forward.
    accumulated_rel_l1 = 0.0
    steps_since_full = 0
    total_steps = len(timesteps)
    eps = 1e-8

    for step_idx, t in enumerate(timesteps):
        latent_model_input = pipe.scheduler.scale_model_input(latents, t).to(transformer_dtype)
        timestep = t.expand(latents.shape[0]).to(transformer_dtype)

        # Decide whether to skip this step's transformer forward.
        in_warmup = step_idx < cache_warmup_steps
        is_last = step_idx == total_steps - 1
        should_skip = False
        skip_reason = "full"
        rel_l1_value = 0.0

        if effective_mode == "fixed":
            should_skip = (
                cache_skip_every >= 2
                and step_idx >= cache_warmup_steps
                and cached_noise_pred is not None
                and (step_idx - cache_warmup_steps) % cache_skip_every != 0
            )
            skip_reason = "fixed_skip" if should_skip else "fixed_full"
        elif effective_mode == "adaptive":
            # Always-full conditions first.
            if in_warmup or is_last or cached_noise_pred is None:
                should_skip = False
                skip_reason = "warmup" if in_warmup else "last_step"
                if cached_noise_pred is None and not in_warmup:
                    skip_reason = "no_cache_yet"
            elif (
                cache_force_full_every > 0
                and steps_since_full >= cache_force_full_every
            ):
                should_skip = False
                skip_reason = "force_floor"
            else:
                # TeaCache-style: relative L1 change of the timestep-conditioned
                # input vs. the input at the last full forward. We use a v0
                # *identity* rescaler — accumulate raw rel_l1 until it crosses
                # the threshold, at which point we force a full forward and
                # reset. The paper's polynomial rescaler can be slotted here
                # offline; until then this is a sound conservative default.
                assert last_full_input is not None
                diff = (latent_model_input - last_full_input).abs().mean()
                denom = last_full_input.abs().mean().clamp_min(eps)
                rel_l1_value = float((diff / denom).detach().cpu())
                accumulated_rel_l1 += rel_l1_value
                if accumulated_rel_l1 < cache_adaptive_threshold:
                    should_skip = True
                    skip_reason = "adaptive_skip"
                else:
                    should_skip = False
                    skip_reason = "accumulator_over_threshold"

        if should_skip:
            noise_pred = cached_noise_pred
            steps_since_full += 1
            if stats is not None:
                stats.skipped += 1
        else:
            if cfg_batched:
                batched_input = latent_model_input.repeat(2, 1, 1, 1, 1)
                batched_timestep = timestep.repeat(2)
                noise_pred = pipe.transformer(
                    hidden_states=batched_input,
                    timestep=batched_timestep,
                    encoder_hidden_states=encoder_pair,
                    fps=fps,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
            else:
                noise_pred_cond = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                noise_pred_uncond = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    fps=fps,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                noise_pred = torch.cat([noise_pred_uncond, noise_pred_cond], dim=0)
            cached_noise_pred = noise_pred
            # Adaptive bookkeeping: only update last_full_input after a real
            # forward; reset accumulator and the periodic-floor counter.
            if effective_mode == "adaptive":
                last_full_input = latent_model_input
                accumulated_rel_l1 = 0.0
                steps_since_full = 0
            if stats is not None:
                stats.full_forwards += 1

        if stats is not None:
            stats.rel_l1_history.append(rel_l1_value)
            stats.skip_reasons.append(skip_reason)

        sample = torch.cat([latents, latents], dim=0)

        # First scheduler call: returns pred_original_sample (x0).
        noise_pred = pipe.scheduler.step(noise_pred, t, sample, return_dict=False)[1]
        pipe.scheduler._step_index -= 1

        # Apply CFG on x0.
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Second scheduler call: actually advance latents.
        latents = pipe.scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=False,
            pred_original_sample=noise_pred,
        )[0]

    # 6. Latent un-normalization + VAE decode + postprocess (reference path).
    if pipe.vae.config.latents_mean is not None:
        latents_mean = pipe.vae.config.latents_mean
        latents_std = pipe.vae.config.latents_std
        latents_mean = (
            torch.tensor(latents_mean)
            .view(1, pipe.vae.config.latent_channels, -1, 1, 1)[:, :, : latents.size(2)]
            .to(latents)
        )
        latents_std = (
            torch.tensor(latents_std)
            .view(1, pipe.vae.config.latent_channels, -1, 1, 1)[:, :, : latents.size(2)]
            .to(latents)
        )
        latents = latents * latents_std / pipe.scheduler.config.sigma_data + latents_mean
    else:
        latents = latents / pipe.scheduler.config.sigma_data

    video = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type=output_type)
