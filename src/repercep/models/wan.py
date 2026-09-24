"""Wan-2.2 engine — Repercep's second world-model family.

Wan-2.2 (Alibaba's "Wan-Video" team, released 2025-07-28) is the second
diffusion-video family Repercep supports, alongside Cosmos-Predict-7B. Its job in
the Phase-2 plan is to prove Repercep's runtime is world-model-native rather than
Cosmos-specific: the same backend Protocol, the same ``WorldModelEngine``
contract, a different transformer family + VAE.

The lead variant is ``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` — a Mixture-of-Experts
text-to-video model with two ~14B-parameter expert transformers (~27B total,
~14B active per step). Apache 2.0 licensed. BF16 weights are ~52 GiB, which
fits comfortably on a 192 GiB MI300X VF without offloading. Diffusers's
``WanPipeline`` exposes a single ``__call__`` that drives both experts; the
secondary guidance scale (``guidance_scale_2``) applies to the second-stage
denoise inside the MoE.

This engine mirrors :class:`repercep.models.cosmos.CosmosEngine` step-for-step
(lazy load, ``EngineInfo``, frame-yielding ``generate``) and pulls the same
``_as_frame_tensor`` shape contract: diffusers' ``output_type='pt'`` yields
``(T, C, H, W)`` in ``[0, 1]``, which we normalize to ``(T, H, W, 3)`` uint8
for ``Frame.pixels``.

Differences from the Cosmos engine, all forced by Wan's pipeline shape:

- **No in-pipeline safety guardrail.** Wan ships none; nothing to neutralize.
- **VAE dtype is FP32, transformer is BF16.** Wan's diffusers example loads the
  VAE separately with ``torch_dtype=torch.float32`` because the VAE is
  numerically sensitive; loading it BF16 produces visible banding.
- **No ``fps`` argument to the pipeline.** Wan was trained at a native 16 FPS
  and the pipeline does not accept an FPS override. We track the requested
  ``params.fps`` for the downstream writer only.
- **Two guidance scales.** The MoE has a second-stage guidance (``guidance_2``),
  exposed as a ``WanConfig`` field with the model-card default (3.0).
- **No native loop yet.** ``cache_skip_every`` / ``use_native_loop`` exist on
  the config as forward-compat hooks but are unused — diffusers' own
  ``WanPipeline.__call__`` is the only path for this first cut. Plugging the
  same denoise + step-skip caching pattern in for Wan is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from repercep.runtime.engine import EngineInfo
from repercep.runtime.types import Frame

if TYPE_CHECKING:
    from collections.abc import Iterator

    from repercep.backend.protocol import Backend
    from repercep.runtime.types import GenerationRequest

#: The lead Wan-2.2 T2V variant — 27B-param MoE, ~14B active per step. Apache 2.0.
DEFAULT_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

#: A smaller Apache-2.0 Wan-2.2 variant — useful for VRAM-constrained smoke tests.
SMALL_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

#: Wan-2.2 was trained at 16 FPS. The pipeline doesn't accept an FPS override
#: at inference; we record this for the downstream video writer.
NATIVE_FPS = 16


@dataclass(slots=True)
class WanConfig:
    """Load-time configuration for :class:`WanEngine`."""

    repo_id: str = DEFAULT_REPO
    device_index: int = 0
    # Transformer compute dtype. Wan's reference path is BF16, native on CDNA3.
    dtype: str = "bfloat16"
    # The Wan VAE is numerically sensitive — Wan-Video's reference loads it as
    # FP32 even when the transformer is BF16; BF16 VAE produces visible banding.
    vae_dtype: str = "float32"
    # Secondary guidance for the MoE A14B variant's second-stage denoise.
    # 3.0 is the Wan2.2-T2V-A14B model-card default. Ignored on non-MoE
    # variants (e.g. TI2V-5B); WanPipeline accepts and discards it there.
    guidance_scale_2: float = 3.0
    # torch.compile the DiT transformer (inductor + triton-rocm). Mirrors the
    # Cosmos engine's flag; first-run compilation is slow, steady-state faster.
    compile_transformer: bool = False
    # Forward-compat hooks: a native Wan loop with CFG batching / step caching
    # is a Phase-2 follow-up. Today the engine always uses the diffusers
    # pipeline ``__call__``. See ``WanEngine.generate``.
    use_native_loop: bool = False
    cache_skip_every: int = 0
    cache_warmup_steps: int = 4
    # Decode the VAE in spatial tiles. Drops peak VRAM by ~6-8 GiB at
    # 1280x720 (the per-frame catenations in AutoencoderKLWan otherwise
    # spike past 80 GiB on H100 when both 14B MoE experts are resident).
    # Output is bit-identical aside from the tile-seam blending the
    # decoder already does internally.
    vae_tiling: bool = False


class WanEngine:
    """Wan-2.2 T2V, served on a Repercep backend.

    Lazy-loaded: constructing the engine is cheap; the ~52 GiB of weights are
    only touched on the first ``load()`` or ``generate()`` call. The diffusers
    pipeline's MoE architecture loads both expert transformers; only one is
    active per denoising step but both must be resident for the per-step
    boundary swap.
    """

    model_name = "wan-2.2-t2v-a14b"

    def __init__(self, backend: Backend, config: WanConfig | None = None) -> None:
        self._backend = backend
        self._config = config if config is not None else WanConfig()
        self._pipe: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def is_compiled(self) -> bool:
        return self._pipe is not None and self._config.compile_transformer

    @property
    def pipeline(self) -> Any:
        """The underlying diffusers pipeline. Raises if not yet loaded."""
        if self._pipe is None:
            raise RuntimeError("engine not loaded; call load() first")
        return self._pipe

    def load(self) -> None:
        """Load the pipeline. Idempotent.

        Wan's reference path loads the VAE separately with FP32 weights and
        passes it into ``WanPipeline.from_pretrained``; the transformer and
        text encoder come down in the configured ``dtype`` (BF16 by default).
        Note: Wan ships no in-pipeline safety checker, so unlike the Cosmos
        engine there is nothing to neutralize at load time.
        """
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline

        compute_dtype = getattr(torch, self._config.dtype)
        vae_dtype = getattr(torch, self._config.vae_dtype)
        device = self._backend.torch_device(self._config.device_index)

        # diffusers is optional ([models] extra) and unresolved in the base
        # mypy environment, so these read as Any rather than untyped calls —
        # no ignore needed here. If a future stub package makes diffusers
        # fully typed, these may need `type: ignore[no-untyped-call]` again.
        vae = AutoencoderKLWan.from_pretrained(
            self._config.repo_id, subfolder="vae", torch_dtype=vae_dtype
        )
        pipe = WanPipeline.from_pretrained(
            self._config.repo_id, vae=vae, torch_dtype=compute_dtype
        )
        pipe.to(device)
        if self._config.vae_tiling:
            pipe.vae.enable_tiling()
        from repercep.attention.wan_processor import maybe_install_repercep_wan_attention

        maybe_install_repercep_wan_attention(pipe)
        if self._config.compile_transformer:
            pipe.transformer = torch.compile(pipe.transformer)
            # MoE A14B variants ship a second transformer for low-noise steps.
            if getattr(pipe, "transformer_2", None) is not None:
                pipe.transformer_2 = torch.compile(pipe.transformer_2)
        self._pipe = pipe

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name=self.model_name,
            backend=self._backend.name,
            device=f"{self._backend.name}:{self._config.device_index}",
            dtype=self._config.dtype,
            ready=self.is_loaded,
        )

    def generate(self, request: GenerationRequest) -> Iterator[Frame]:
        import torch

        self.load()
        assert self._pipe is not None
        params = request.params

        generator: torch.Generator | None = None
        if params.seed is not None:
            device = self._backend.torch_device(self._config.device_index)
            generator = torch.Generator(device=device).manual_seed(params.seed)

        # Native loop is a Phase-2 follow-up for Wan; today we always drive the
        # diffusers ``WanPipeline.__call__``. The ``use_native_loop`` and
        # ``cache_skip_every`` knobs on WanConfig exist so the engine config
        # shape is stable across model families — they're inert until the
        # Wan-native loop lands. params.fps is not passed: Wan was trained at
        # 16 FPS and the pipeline does not accept an FPS override.
        pipe_kwargs: dict[str, Any] = dict(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=params.height,
            width=params.width,
            num_frames=params.num_frames,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
            output_type="pt",
        )
        # guidance_scale_2 is the MoE A14B second-stage scale. Non-MoE Wan
        # variants (e.g. TI2V-5B) raise if it's passed — they have no boundary
        # between high-noise / low-noise experts. Probe the pipeline config
        # rather than the repo_id so any future MoE/non-MoE variant works.
        if getattr(self._pipe.config, "boundary_ratio", None) is not None:
            pipe_kwargs["guidance_scale_2"] = self._config.guidance_scale_2
        output = self._pipe(**pipe_kwargs)
        video = _as_frame_tensor(output.frames[0])

        total = int(video.shape[0])
        for index in range(total):
            yield Frame(index=index, total=total, pixels=video[index])


def _as_frame_tensor(video: Any) -> Any:
    """Normalize a diffusers video output to ``(T, H, W, 3)`` uint8 on CPU.

    Mirrors :func:`repercep.models.cosmos._as_frame_tensor`. The Wan pipeline's
    ``output_type='pt'`` yields ``(T, C, H, W)`` in ``[0, 1]``, same shape
    contract as Cosmos.
    """
    import torch

    tensor = video if isinstance(video, torch.Tensor) else torch.as_tensor(video)
    tensor = tensor.detach().to("cpu", dtype=torch.float32)
    if tensor.ndim != 4:
        raise ValueError(f"unexpected video tensor rank: {tuple(tensor.shape)}")
    # diffusers `output_type='pt'` yields (T, C, H, W); tolerate a channels-last
    # (T, H, W, C) layout too.
    if tensor.shape[1] in (1, 3):
        tensor = tensor.permute(0, 2, 3, 1)
    return (tensor.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
