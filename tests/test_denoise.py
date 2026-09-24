"""Regression tests for ``repercep.runtime.denoise``.

These are structural tests — they don't need a GPU or model — but they pin the
invariants that recently caused observable runtime OOMs.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from repercep.runtime import denoise


def test_denoise_cosmos_video_runs_under_no_grad_gate() -> None:
    """Without ``torch.inference_mode()`` (or ``no_grad``) wrapping the body,
    the autograd graph holds activations across all diffusion steps. At 17f /
    8 steps that fits in HBM (~28 GiB peak observed); at 121f / 36 steps it
    OOMs at ~189 GiB on a 192 GiB MI300X, which is the regression this guard
    prevents. Diffusers' own ``CosmosTextToWorldPipeline.__call__`` has the
    same decorator — this is parity, not a workaround.
    """
    src = inspect.getsource(denoise.denoise_cosmos_video)
    assert "inference_mode" in src or "no_grad" in src, (
        "denoise_cosmos_video must wrap its body in torch.inference_mode() or "
        "torch.no_grad(); without it activation memory grows ~linearly in "
        "step count and 121f / 36 steps OOMs."
    )


def test_denoise_rejects_unknown_cache_mode() -> None:
    """``cache_mode`` must be one of the documented set."""
    with pytest.raises(ValueError, match="cache_mode"):
        denoise.denoise_cosmos_video(
            pipe=object(),
            prompt="ignored",
            cache_mode="bogus",
        )


def test_denoise_stats_initial_state() -> None:
    """``DenoiseStats`` is a plain accumulator — start zero, lists empty."""
    stats = denoise.DenoiseStats()
    assert stats.full_forwards == 0
    assert stats.skipped == 0
    assert stats.rel_l1_history == []
    assert stats.skip_reasons == []


def _make_stub_pipe(steps: int) -> Any:
    """Build a torch-only stub of the diffusers pipeline pieces the loop touches.

    Enough to exercise the loop body without touching CUDA: an identity
    transformer (one batched call), a no-op scheduler, fake encoder/vae,
    everything in fp32 on CPU. The real values are not meaningful — the test
    only checks the *control flow* (full vs. skipped forwards under each
    caching mode).
    """
    import torch

    class _Scheduler:
        def __init__(self) -> None:
            self.timesteps = torch.arange(steps, dtype=torch.float32)
            self._step_index = 0
            self.config = type("C", (), {"sigma_data": 1.0})()

        def set_timesteps(self, n: int, device: Any) -> None:
            self.timesteps = torch.arange(n, dtype=torch.float32)

        def scale_model_input(self, latents: Any, t: Any) -> Any:
            # Inject a varying scale so the *input* changes between steps,
            # which is what the adaptive gate measures. Larger t → larger scale.
            return latents * (1.0 + 0.05 * float(t))

        def step(
            self,
            noise_pred: Any,
            t: Any,
            sample: Any,
            return_dict: bool = True,
            pred_original_sample: Any = None,
        ) -> Any:
            # Two-call pattern: first call returns (x0_pred, x0_pred) tuple;
            # second call returns advanced latents.
            self._step_index += 1
            if pred_original_sample is None:
                return (noise_pred, noise_pred)
            # Slight drift each step so latents change over time.
            return (sample * 0.99,)

    class _Transformer:
        def __init__(self) -> None:
            self.call_count = 0
            self.config = type("C", (), {"in_channels": 4})()
            self.dtype = torch.float32

        def __call__(self, **kwargs: Any) -> Any:
            self.call_count += 1
            hidden = kwargs["hidden_states"]
            # Return zeros of the right batched shape (B, C, T, H, W).
            return (torch.zeros_like(hidden),)

    class _VAE:
        def __init__(self) -> None:
            self.config = type(
                "C",
                (),
                {"latents_mean": None, "latents_std": None, "latent_channels": 4},
            )()
            self.dtype = torch.float32

        def decode(self, latents: Any, return_dict: bool = True) -> Any:
            # Pass through reshaped as (B, C, T, H, W) → (T, C, H, W).
            return (latents,)

    class _VideoProcessor:
        def postprocess_video(self, video: Any, output_type: str) -> Any:
            return video

    class _Pipe:
        def __init__(self) -> None:
            self._execution_device = "cpu"
            self.transformer = _Transformer()
            self.scheduler = _Scheduler()
            self.vae = _VAE()
            self.video_processor = _VideoProcessor()

        def encode_prompt(
            self,
            prompt: str,
            negative_prompt: Any,
            do_classifier_free_guidance: bool,
            device: Any,
            dtype: Any,
        ) -> Any:
            embed = torch.zeros(1, 4, dtype=dtype)
            return embed, embed

        def prepare_latents(
            self,
            batch_size: int,
            num_channels_latents: int,
            height: int,
            width: int,
            num_frames: int,
            dtype: Any,
            device: Any,
            generator: Any,
            latents: Any,
        ) -> Any:
            return torch.randn(
                batch_size, num_channels_latents, 2, 2, 2, dtype=dtype
            )

    return _Pipe()


def test_adaptive_cache_logs_full_and_skipped() -> None:
    """Adaptive cache must:
    - Always run the warmup steps as full forwards (no skipping).
    - Always run the last step as a full forward (quality floor).
    - Account every step into either ``full_forwards`` or ``skipped``.
    """
    steps = 12
    pipe = _make_stub_pipe(steps)
    stats = denoise.DenoiseStats()
    denoise.denoise_cosmos_video(
        pipe,
        prompt="ignored",
        height=8,
        width=8,
        num_frames=1,
        num_inference_steps=steps,
        cache_mode="adaptive",
        cache_adaptive_threshold=0.1,
        cache_warmup_steps=4,
        cache_force_full_every=8,
        stats=stats,
        # The stub doesn't implement CFG-batched transformer shapes, so use the
        # sequential branch for control-flow checking.
        cfg_batched=False,
    )
    assert stats.full_forwards + stats.skipped == steps
    # First cache_warmup_steps must all be full forwards.
    assert all(r in ("warmup", "fixed_full", "full") for r in stats.skip_reasons[:4])
    # Last step must be a full forward.
    assert stats.skip_reasons[-1] in ("last_step", "fixed_full", "full", "force_floor")


def test_fixed_cache_preserves_legacy_cadence() -> None:
    """With ``cache_mode="fixed"`` and ``cache_skip_every=4``, post-warmup we
    expect 1-in-4 full forwards + 3-in-4 skips — exactly the F16/F17 cadence
    landed in the BUILD_LOG. With 4 warmup + 12 post-warmup steps and skip=4,
    that's 4 + ceil(12/4)=4+3=7 full forwards plus 12-3=9 skips? Actually 12/4
    = 3 skipped windows of size 4: position 0 (modulo) = full, 1,2,3 = skip.
    Let's count: warmup [0,1,2,3]=full(4); then [4..15]=12 steps: at offsets
    0,4,8 = full (3); the remaining 9 are skipped. Total: 7 full, 9 skipped.
    """
    steps = 16
    pipe = _make_stub_pipe(steps)
    stats = denoise.DenoiseStats()
    denoise.denoise_cosmos_video(
        pipe,
        prompt="ignored",
        height=8,
        width=8,
        num_frames=1,
        num_inference_steps=steps,
        cache_mode="fixed",
        cache_skip_every=4,
        cache_warmup_steps=4,
        stats=stats,
        cfg_batched=False,
    )
    assert stats.full_forwards == 7
    assert stats.skipped == 9


def test_no_cache_mode_runs_full_every_step() -> None:
    """``cache_mode="none"`` must produce one full forward per step."""
    steps = 6
    pipe = _make_stub_pipe(steps)
    stats = denoise.DenoiseStats()
    denoise.denoise_cosmos_video(
        pipe,
        prompt="ignored",
        height=8,
        width=8,
        num_frames=1,
        num_inference_steps=steps,
        cache_mode="none",
        stats=stats,
        cfg_batched=False,
    )
    assert stats.full_forwards == steps
    assert stats.skipped == 0


def test_legacy_cache_skip_every_backward_compat() -> None:
    """Backward compat: passing ``cache_skip_every>=2`` with the default
    ``cache_mode="none"`` must still get the legacy fixed cadence.
    """
    steps = 8
    pipe = _make_stub_pipe(steps)
    stats = denoise.DenoiseStats()
    denoise.denoise_cosmos_video(
        pipe,
        prompt="ignored",
        height=8,
        width=8,
        num_frames=1,
        num_inference_steps=steps,
        # Note: cache_mode is the default "none" — but skip_every=2 should
        # fall through to "fixed" for backward compatibility.
        cache_skip_every=2,
        cache_warmup_steps=4,
        stats=stats,
        cfg_batched=False,
    )
    # 4 warmup full + 4 post-warmup with skip=2 (full at offset 0, 2 = 2 full,
    # skip at 1, 3 = 2 skips) → 6 full, 2 skips.
    assert stats.full_forwards == 6
    assert stats.skipped == 2
