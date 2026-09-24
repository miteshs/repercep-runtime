"""Cosmos-Predict-7B engine.

Wraps the HuggingFace ``diffusers`` ``CosmosTextToWorldPipeline`` as a Repercep
``WorldModelEngine``. The diffusers path is deliberate (ADR-0002 and the Cosmos
research brief): unlike NVIDIA's reference ``cosmos-predict1`` repo it has no
TransformerEngine or apex dependency, so it runs on ROCm PyTorch with attention
served by ``scaled_dot_product_attention`` — which itself dispatches to
aotriton flash kernels on MI300X.

Safety guardrail: the diffusers pipeline force-registers a ``CosmosSafetyChecker``
as a pipeline component, which does not compose with the current diffusers
device detection. Repercep instead neutralizes the in-pipeline checker and, when
``enable_guardrail`` is set, runs ``cosmos_guardrail`` itself around generation
— a text check before, a video face-blur check after.

Generation is not yet frame-streaming: the diffusion loop denoises the whole
latent volume and the VAE decodes it in one shot, so ``generate`` runs the
pipeline to completion and then yields the decoded frames. Denoise-time
streaming is a later optimization (implementation plan, Phase 2).
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

#: The diffusers-format Cosmos-Predict1 7B Text2World repository.
DEFAULT_REPO = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"

#: The Cosmos safety guardrail asset repository.
GUARDRAIL_REPO = "nvidia/Cosmos-1.0-Guardrail"


class GuardrailError(RuntimeError):
    """Raised when the Cosmos safety guardrail rejects a prompt or a video."""


@dataclass(slots=True)
class CosmosConfig:
    """Load-time configuration for :class:`CosmosEngine`."""

    repo_id: str = DEFAULT_REPO
    device_index: int = 0
    dtype: str = "bfloat16"  # reference dtype; native on CDNA3
    # The NVIDIA Open Model License requires the guardrail for deployment. When
    # True, Repercep runs cosmos_guardrail around generation (text + video).
    enable_guardrail: bool = False
    # torch.compile the DiT transformer (inductor + triton-rocm). First-run
    # compilation is slow; steady-state is faster. See docs/OPTIMIZATION.md.
    compile_transformer: bool = False
    # torch.compile mode passed to ``torch.compile`` when ``compile_transformer``
    # is True. ``None`` is the inductor default; ``"reduce-overhead"`` cuts
    # Python overhead with CUDA graphs; ``"max-autotune"`` is the slowest
    # compile but fastest steady-state. Surfaced as a knob mainly because the
    # inductor + triton-rocm stack has been seen to segfault at 121f shapes
    # in default mode (BUILD_LOG F15 / F18) — alternative modes are the
    # in-tree workaround until upstream lands a fix.
    compile_mode: str | None = None
    # If True, pass ``dynamic=True`` to ``torch.compile`` so a single compiled
    # graph handles multiple input shapes. Slightly slower per call than a
    # static-shape compile, but avoids per-shape recompiles and (per F18)
    # sidesteps the 121-f inductor-rocm segfault path. Default False.
    compile_dynamic: bool = False
    # Safety gate: if the request's ``num_frames`` exceeds this threshold and
    # ``compile_transformer`` is True, ``CosmosEngine`` falls back to the
    # uncompiled DiT and emits a clear message. 49 is the empirical floor —
    # 49f compiles cleanly today, 121f segfaults mid-warmup (F15). Set to 0
    # to disable the gate.
    compile_max_frames: int = 64
    # Use Repercep's native denoising loop in place of the diffusers
    # `CosmosTextToWorldPipeline.__call__`. Enables CFG batching (one batch-2
    # transformer forward per step instead of two batch-1 forwards).
    # See `repercep.runtime.denoise` and docs/OPTIMIZATION.md.
    use_native_loop: bool = False
    # Step-skip caching in the native loop: after `cache_warmup_steps`, run a
    # full DiT forward only every Nth step and reuse the cached output for the
    # rest. 0 disables. Real work reduction; quality dial.
    cache_skip_every: int = 0
    cache_warmup_steps: int = 4
    # Caching strategy. ``"none"`` disables caching entirely (the default).
    # ``"fixed"`` preserves the legacy ``cache_skip_every`` cadence behaviour.
    # ``"adaptive"`` selects the TeaCache-style input-similarity gate — a step
    # is skipped only when the relative L1 change of the model input vs. the
    # last full forward is below ``cache_adaptive_threshold``.
    cache_mode: str = "none"
    # Adaptive-cache (``cache_mode="adaptive"``) threshold on the *accumulated*
    # relative L1 distance of the timestep-conditioned latent input. Lower is
    # more conservative (fewer skips, higher quality). v0 uses an identity
    # rescaler — accumulating raw `rel_l1` until it crosses this threshold.
    # 0.3 was tuned on 121f/36 step: it beats the fixed `skip=4` headline
    # (152.9 s vs 175 s here, 11 vs 12 full forwards), with motion stat
    # matching fixed within noise. See ``scripts/bench_caching.py``.
    cache_adaptive_threshold: float = 0.3
    # Adaptive cache only: force a full forward at least every N steps even if
    # the gate would skip. Acts as a quality floor against pathologically
    # static prefixes. ``0`` disables the periodic full-forward floor.
    # 16 matches the cadence that the tuned ``threshold=0.3`` actually hits
    # at 121f/36 step — bigger and the floor never kicks in; smaller and it
    # overrides the gate.
    cache_force_full_every: int = 16


class CosmosEngine:
    """Cosmos-Predict1-7B Text2World, served on a Repercep backend.

    The pipeline is loaded lazily — constructing a ``CosmosEngine`` is cheap;
    the ~38 GB of weights are only touched on the first ``load()`` or
    ``generate()`` call.
    """

    model_name = "cosmos-predict1-7b-text2world"

    def __init__(self, backend: Backend, config: CosmosConfig | None = None) -> None:
        self._backend = backend
        self._config = config if config is not None else CosmosConfig()
        self._pipe: Any | None = None
        self._guardrail: Any | None = None
        # When ``compile_transformer`` is set and the frame-size gate may need
        # to fall back to the uncompiled DiT (F15 / F18), we keep a handle on
        # the original transformer module so ``generate()`` can swap it in.
        self._uncompiled_transformer: Any | None = None
        # Set to True if the compile gate has been tripped at least once.
        self._compile_gated: bool = False

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
        """Load the pipeline (and, if enabled, the guardrail). Idempotent."""
        if self._pipe is not None:
            return
        import torch
        from diffusers import CosmosTextToWorldPipeline

        # Repercep always neutralizes the in-pipeline safety checker and runs the
        # guardrail itself around generation — see the module docstring.
        _disable_cosmos_guardrail()

        # Bridge ``REPERCEP_FP8_ATTENTION`` to diffusers' attention dispatcher.
        # Cosmos's ``CosmosAttnProcessor2_0`` calls
        # ``diffusers.models.attention_dispatch.dispatch_attention_fn``, which
        # never consults ``repercep.attention.select_attention_op``. The bridge
        # in ``repercep.attention.diffusers_backend`` registers a ``"repercep_fp8"``
        # backend with that dispatcher; flipping ``REPERCEP_FP8_ATTENTION`` here
        # selects it for the rest of the process. See BUILD_LOG F19 and the
        # Phase 2.5 entry.
        from repercep.attention.diffusers_backend import maybe_activate_from_env

        maybe_activate_from_env()

        dtype = getattr(torch, self._config.dtype)
        device = self._backend.torch_device(self._config.device_index)
        # diffusers is optional ([models] extra) and unresolved in the base
        # mypy environment, so this reads as Any rather than an untyped call —
        # no ignore needed here. If a future stub package makes diffusers
        # fully typed, this may need `type: ignore[no-untyped-call]` again.
        pipe = CosmosTextToWorldPipeline.from_pretrained(
            self._config.repo_id, torch_dtype=dtype
        )
        pipe.to(device)
        if self._config.compile_transformer:
            # F15 / F18: at 121-frame shapes the inductor + triton-rocm path
            # has been observed to segfault mid-warmup. ``compile_mode`` and
            # ``compile_dynamic`` are the two in-tree knobs to try
            # (``mode="reduce-overhead"`` and ``dynamic=True`` are the two
            # workarounds most commonly cited for shape-specific Triton-rocm
            # bugs). The frame-size gate at ``generate()`` is the belt-and-
            # braces fallback that just disables compile above the safe floor.
            self._uncompiled_transformer = pipe.transformer
            compile_kwargs: dict[str, Any] = {}
            if self._config.compile_mode is not None:
                compile_kwargs["mode"] = self._config.compile_mode
            if self._config.compile_dynamic:
                compile_kwargs["dynamic"] = True
            pipe.transformer = torch.compile(pipe.transformer, **compile_kwargs)
        self._pipe = pipe

        if self._config.enable_guardrail:
            self._guardrail = _load_guardrail(device)

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name=self.model_name,
            backend=self._backend.name,
            device=f"{self._backend.name}:{self._config.device_index}",
            dtype=self._config.dtype,
            ready=self.is_loaded,
        )

    def _gate_compile_if_needed(self, num_frames: int) -> None:
        """Fall back to the uncompiled DiT if ``num_frames`` exceeds the
        F15-known-safe ceiling and a fallback module is available.

        F15 / F18: ``torch.compile`` (inductor + triton-rocm 7.2) segfaults
        mid-warmup at 121-frame shapes; 49 frames compiles cleanly. The
        ``compile_max_frames`` config exposes the threshold; ``0`` disables
        the gate (caller opts out — useful once upstream lands a fix).
        """
        if self._pipe is None or self._uncompiled_transformer is None:
            return
        ceiling = self._config.compile_max_frames
        if ceiling <= 0:
            return
        if num_frames <= ceiling:
            return
        # First-time fallback for this engine: log it once.
        if not self._compile_gated:
            print(
                f"[repercep] compile gate tripped: num_frames={num_frames} > "
                f"compile_max_frames={ceiling}; falling back to the uncompiled "
                "DiT for this and subsequent calls (BUILD_LOG F15/F18).",
                flush=True,
            )
            self._compile_gated = True
        # Swap in the original (uncompiled) transformer for this request.
        # Idempotent: if already swapped, this is a no-op.
        if self._pipe.transformer is not self._uncompiled_transformer:
            self._pipe.transformer = self._uncompiled_transformer

    def generate(self, request: GenerationRequest) -> Iterator[Frame]:
        import torch

        self.load()
        assert self._pipe is not None
        params = request.params

        if self._guardrail is not None and not self._guardrail.check_text_safety(request.prompt):
            raise GuardrailError("prompt rejected by the Cosmos safety guardrail")

        # F15 / F18 safety gate. If the request exceeds the frame ceiling for
        # which inductor + triton-rocm is known to compile cleanly, swap the
        # compiled DiT for the original uncompiled module for this request.
        # The compile cost has already been paid (in load()) but is not used —
        # better that than a SIGSEGV mid-warmup of the 121-f reference config.
        self._gate_compile_if_needed(params.num_frames)

        generator: torch.Generator | None = None
        if params.seed is not None:
            device = self._backend.torch_device(self._config.device_index)
            generator = torch.Generator(device=device).manual_seed(params.seed)

        if self._config.use_native_loop:
            from repercep.runtime.denoise import denoise_cosmos_video

            video_raw = denoise_cosmos_video(
                self._pipe,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=params.height,
                width=params.width,
                num_frames=params.num_frames,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                fps=params.fps,
                seed=params.seed,
                output_type="pt",
                cache_skip_every=self._config.cache_skip_every,
                cache_warmup_steps=self._config.cache_warmup_steps,
                cache_mode=self._config.cache_mode,
                cache_adaptive_threshold=self._config.cache_adaptive_threshold,
                cache_force_full_every=self._config.cache_force_full_every,
            )
            video = _as_frame_tensor(video_raw[0])
        else:
            output = self._pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=params.height,
                width=params.width,
                num_frames=params.num_frames,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                fps=params.fps,
                generator=generator,
                output_type="pt",
            )
            video = _as_frame_tensor(output.frames[0])

        if self._guardrail is not None:
            video = _apply_video_guardrail(self._guardrail, video)

        total = int(video.shape[0])
        for index in range(total):
            yield Frame(index=index, total=total, pixels=video[index])


def _disable_cosmos_guardrail() -> None:
    """Neutralize the diffusers pipeline's built-in safety checker.

    ``CosmosTextToWorldPipeline.__init__`` force-constructs a
    ``CosmosSafetyChecker`` and registers it as a pipeline component — which
    both pulls the heavy ``cosmos_guardrail`` stack into pipeline construction
    and (with current diffusers) breaks the pipeline's device detection. Repercep
    always swaps it for a no-op and, when the guardrail is enabled, runs
    ``cosmos_guardrail`` itself around generation (see ``_load_guardrail`` and
    ``CosmosEngine.generate``).
    """
    from diffusers.pipelines.cosmos import pipeline_cosmos_text2world as mod

    class _DisabledGuardrail:
        def to(self, *args: object, **kwargs: object) -> _DisabledGuardrail:
            return self

        def check_text_safety(self, prompt: object) -> bool:
            return True

        def check_video_safety(self, video: Any) -> Any:
            return video

    mod.CosmosSafetyChecker = _DisabledGuardrail


def _ensure_guardrail_assets() -> None:
    """Pre-fetch the Cosmos guardrail assets and fix up their NLTK corpora.

    ``cosmos_guardrail`` 0.3.0 fetches its assets with
    ``snapshot_download(..., allow_patterns=["blocklist"])``; against current
    ``huggingface_hub`` that bare pattern matches no files, so the guardrail
    loads from an empty directory and crashes. Pre-fetching the blocklist and
    face-blur trees into the shared snapshot directory makes the package's
    narrowed calls resolve. The ~13 GB legacy ``aegis/`` cache in that repo is
    deliberately skipped — 0.3.0 uses Qwen3Guard, not Llama-Guard.
    """
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        GUARDRAIL_REPO, allow_patterns=["blocklist/*", "face_blur_filter/*"]
    )
    _materialize_nltk_data(snapshot)


def _materialize_nltk_data(snapshot_dir: str) -> None:
    """Copy the guardrail's NLTK corpora to a real, non-symlinked directory.

    NLTK 3.9 refuses to read corpus files whose paths resolve into the
    HuggingFace blob cache ("Security Violation: Unauthorized path"), which
    silently disables the blocklist's lemmatized keyword matching. Copying the
    corpora out of the symlinked snapshot and prepending the copy to
    ``nltk.data.path`` restores it.
    """
    import shutil
    from pathlib import Path

    import nltk

    source = Path(snapshot_dir) / "blocklist" / "nltk_data"
    target = Path.home() / ".cache" / "repercep" / "nltk_data"
    if source.is_dir() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, symlinks=False)
    entry = str(target)
    if target.is_dir() and entry not in nltk.data.path:
        nltk.data.path.insert(0, entry)


def _load_guardrail(device: Any) -> Any:
    """Construct the Cosmos safety guardrail and move it onto ``device``.

    cosmos_guardrail 0.3.0 uses a word blocklist plus Qwen3Guard-Gen-0.6B for
    text safety and a RetinaFace face-blur postprocessor for video.
    """
    _ensure_guardrail_assets()
    from cosmos_guardrail import CosmosSafetyChecker

    guardrail = CosmosSafetyChecker()
    guardrail.to(device)
    return guardrail


def _apply_video_guardrail(guardrail: Any, video: Any) -> Any:
    """Run the guardrail's video check (face blur); raise if it blocks."""
    import numpy as np
    import torch

    checked = guardrail.check_video_safety(video.cpu().numpy())
    if checked is None:
        raise GuardrailError("generated video rejected by the Cosmos safety guardrail")
    array = np.asarray(checked)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return torch.from_numpy(array)


def _as_frame_tensor(video: Any) -> Any:
    """Normalize a diffusers video output to ``(T, H, W, 3)`` uint8 on CPU."""
    import torch

    tensor = video if isinstance(video, torch.Tensor) else torch.as_tensor(video)
    tensor = tensor.detach().to("cpu", dtype=torch.float32)
    if tensor.ndim != 4:
        raise ValueError(f"unexpected video tensor rank: {tuple(tensor.shape)}")
    # diffusers `output_type='pt'` yields (T, C, H, W) in [0, 1]; tolerate
    # a channels-last (T, H, W, C) layout too.
    if tensor.shape[1] in (1, 3):
        tensor = tensor.permute(0, 2, 3, 1)
    return (tensor.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
