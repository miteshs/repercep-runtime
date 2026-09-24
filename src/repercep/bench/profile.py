"""Per-stage profiling for the Cosmos and Wan pipelines.

``repercep.bench.harness`` times an engine end-to-end. This module splits one
generation into its stages — text encode, the DiT denoising loop, VAE decode,
and everything else — so optimization effort lands where the time actually is
(see ``docs/OPTIMIZATION.md``).

Text-encode and DiT stages are timed with PyTorch forward hooks, which fire
outside the compiled graph and so are valid whether or not the DiT has been
``torch.compile``-d. VAE decode is timed by wrapping ``vae.decode`` (a method,
not a ``forward``). Every timer CUDA-synchronizes, so the numbers are real
device time, not kernel-launch time.

The Wan profiler additionally accounts for the MoE second expert
(``transformer_2``): the high-noise expert and the low-noise expert each get
their own counter, summed into ``dit_loop_s`` so the headline number stays
comparable to Cosmos's single-transformer profile.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from repercep.models.cosmos import CosmosEngine
    from repercep.models.wan import WanEngine
    from repercep.runtime.types import GenerationRequest


class CosmosProfile(BaseModel):
    """Per-stage wall-time breakdown of one Cosmos generation."""

    model_config = ConfigDict(extra="forbid")

    total_s: float
    text_encode_s: float
    dit_loop_s: float
    vae_decode_s: float
    other_s: float  # latent prep, scheduler, CFG concat, postprocess, frame copy
    dit_calls: int  # transformer forwards (== 2 per step under classifier-free guidance)
    text_encode_calls: int
    vae_decode_calls: int
    compiled: bool

    @property
    def dit_share(self) -> float:
        return self.dit_loop_s / self.total_s if self.total_s else 0.0


class WanProfile(BaseModel):
    """Per-stage wall-time breakdown of one Wan-2.2 generation.

    Adds two MoE-specific fields not present in :class:`CosmosProfile`:
    ``dit_high_noise_s`` (time in ``transformer``, the high-noise expert that
    handles early steps) and ``dit_low_noise_s`` (time in ``transformer_2``,
    the low-noise expert that handles later steps). ``dit_loop_s`` is the sum
    of the two — same semantics as the Cosmos counterpart so the two reports
    are directly comparable.
    """

    model_config = ConfigDict(extra="forbid")

    total_s: float
    text_encode_s: float
    dit_loop_s: float  # transformer + transformer_2 combined
    dit_high_noise_s: float  # transformer (high-noise expert, early steps)
    dit_low_noise_s: float  # transformer_2 (low-noise expert, later steps)
    vae_decode_s: float
    other_s: float
    dit_calls: int  # transformer forwards (transformer + transformer_2 combined)
    dit_high_noise_calls: int
    dit_low_noise_calls: int
    text_encode_calls: int
    vae_decode_calls: int
    compiled: bool

    @property
    def dit_share(self) -> float:
        return self.dit_loop_s / self.total_s if self.total_s else 0.0


@dataclass
class _Stage:
    seconds: float = 0.0
    calls: int = 0
    start: float = 0.0


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


class _Probe:
    """Installs per-stage timers on a diffusers pipeline for one run.

    Used by both :func:`profile_cosmos` (single ``transformer``) and
    :func:`profile_wan` (MoE: ``transformer`` + ``transformer_2``). If the
    pipeline carries a ``transformer_2`` attribute that is not ``None``, the
    second-expert hook is installed automatically and the ``dit2`` counter
    accumulates separately.
    """

    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe
        self.text = _Stage()
        self.dit = _Stage()
        self.dit2 = _Stage()  # MoE second expert (transformer_2); None on Cosmos
        self.vae = _Stage()
        self._handles: list[Any] = []
        self._vae_decode_original: Any = None
        self._has_transformer_2 = getattr(pipe, "transformer_2", None) is not None

    def __enter__(self) -> _Probe:
        self._hook(self._pipe.text_encoder, self.text)
        self._hook(self._pipe.transformer, self.dit)
        if self._has_transformer_2:
            self._hook(self._pipe.transformer_2, self.dit2)
        self._wrap_vae_decode()
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._vae_decode_original is not None:
            self._pipe.vae.decode = self._vae_decode_original

    def _hook(self, module: Any, stage: _Stage) -> None:
        def pre(_m: Any, _args: Any) -> None:
            _sync()
            stage.start = time.perf_counter()

        def post(_m: Any, _args: Any, _out: Any) -> None:
            _sync()
            stage.seconds += time.perf_counter() - stage.start
            stage.calls += 1

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def _wrap_vae_decode(self) -> None:
        original = self._pipe.vae.decode
        self._vae_decode_original = original
        stage = self.vae

        def timed(*args: Any, **kwargs: Any) -> Any:
            _sync()
            start = time.perf_counter()
            out = original(*args, **kwargs)
            _sync()
            stage.seconds += time.perf_counter() - start
            stage.calls += 1
            return out

        self._pipe.vae.decode = timed


def profile_cosmos(
    engine: CosmosEngine, request: GenerationRequest, *, warmup: int = 1
) -> CosmosProfile:
    """Profile one Cosmos generation, split by stage.

    Args:
        engine: a loaded-or-loadable ``CosmosEngine``.
        request: the generation to profile.
        warmup: untimed generations first — absorbs ROCm kernel autotuning and,
            when the DiT is compiled, the one-time ``torch.compile`` cost.
    """
    engine.load()
    pipe = engine.pipeline

    for _ in range(warmup):
        _drain(engine, request)

    _sync()
    with _Probe(pipe) as probe:
        start = time.perf_counter()
        _drain(engine, request)
        _sync()
        total = time.perf_counter() - start

    accounted = probe.text.seconds + probe.dit.seconds + probe.vae.seconds
    return CosmosProfile(
        total_s=total,
        text_encode_s=probe.text.seconds,
        dit_loop_s=probe.dit.seconds,
        vae_decode_s=probe.vae.seconds,
        other_s=max(0.0, total - accounted),
        dit_calls=probe.dit.calls,
        text_encode_calls=probe.text.calls,
        vae_decode_calls=probe.vae.calls,
        compiled=engine.is_compiled,
    )


def _drain(engine: CosmosEngine, request: GenerationRequest) -> None:
    for _ in engine.generate(request):
        pass


def profile_wan(
    engine: WanEngine, request: GenerationRequest, *, warmup: int = 1
) -> WanProfile:
    """Profile one Wan-2.2 generation, split by stage.

    Mirrors :func:`profile_cosmos` step-for-step but accounts for the MoE
    second expert (``transformer_2``). The high-noise expert handles steps
    where the timestep ``t >= boundary_ratio * num_train_timesteps`` (early
    steps in the denoise schedule); the low-noise expert handles the rest.
    Both timings are reported separately so the cost of the second-expert
    handoff is visible in the breakdown.

    Args:
        engine: a loaded-or-loadable :class:`repercep.models.wan.WanEngine`.
        request: the generation to profile.
        warmup: untimed generations first — absorbs ROCm kernel autotuning and,
            when the DiT is compiled, the one-time ``torch.compile`` cost.
    """
    engine.load()
    pipe = engine.pipeline

    for _ in range(warmup):
        _drain_wan(engine, request)

    _sync()
    with _Probe(pipe) as probe:
        start = time.perf_counter()
        _drain_wan(engine, request)
        _sync()
        total = time.perf_counter() - start

    dit_loop = probe.dit.seconds + probe.dit2.seconds
    accounted = probe.text.seconds + dit_loop + probe.vae.seconds
    return WanProfile(
        total_s=total,
        text_encode_s=probe.text.seconds,
        dit_loop_s=dit_loop,
        dit_high_noise_s=probe.dit.seconds,
        dit_low_noise_s=probe.dit2.seconds,
        vae_decode_s=probe.vae.seconds,
        other_s=max(0.0, total - accounted),
        dit_calls=probe.dit.calls + probe.dit2.calls,
        dit_high_noise_calls=probe.dit.calls,
        dit_low_noise_calls=probe.dit2.calls,
        text_encode_calls=probe.text.calls,
        vae_decode_calls=probe.vae.calls,
        compiled=engine.is_compiled,
    )


def _drain_wan(engine: WanEngine, request: GenerationRequest) -> None:
    for _ in engine.generate(request):
        pass
