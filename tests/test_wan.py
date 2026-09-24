"""Tests for the Wan-2.2 engine.

Construction, identity, defaults, and ``WorldModelEngine`` Protocol conformance
are checked here. Actual generation needs ~52 GiB of downloaded weights and a
GPU, and is exercised by ``scripts/run_wan.py`` rather than the unit suite.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from repercep.backend.cpu import CPUBackend
from repercep.backend.rocm import ROCmBackend
from repercep.models import (
    WAN_DEFAULT_REPO,
    WAN_NATIVE_FPS,
    WAN_SMALL_REPO,
    WanConfig,
    WanEngine,
)
from repercep.runtime.engine import WorldModelEngine


@contextmanager
def _fake_diffusers_loader(fake_vae: Any, fake_pipe: Any) -> Iterator[None]:
    """Provide just enough of diffusers for WanEngine.load unit tests."""

    module = types.ModuleType("diffusers")

    class FakeAutoencoderKLWan:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return fake_vae

    class FakeWanPipeline:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return fake_pipe

    module.__dict__["AutoencoderKLWan"] = FakeAutoencoderKLWan
    module.__dict__["WanPipeline"] = FakeWanPipeline
    prior = sys.modules.get("diffusers")
    sys.modules["diffusers"] = module
    try:
        yield
    finally:
        if prior is None:
            sys.modules.pop("diffusers", None)
        else:
            sys.modules["diffusers"] = prior


@contextmanager
def _fake_wan_processor_module(processor_cls: type[Any]) -> Iterator[None]:
    """Provide the diffusers WanAttnProcessor import path only."""

    module_names = (
        "diffusers",
        "diffusers.models",
        "diffusers.models.transformers",
        "diffusers.models.transformers.transformer_wan",
    )
    prior = {name: sys.modules.get(name) for name in module_names}

    diffusers = types.ModuleType("diffusers")
    diffusers.__dict__["__path__"] = []
    models = types.ModuleType("diffusers.models")
    models.__dict__["__path__"] = []
    transformers = types.ModuleType("diffusers.models.transformers")
    transformers.__dict__["__path__"] = []
    transformer_wan = types.ModuleType("diffusers.models.transformers.transformer_wan")
    transformer_wan.__dict__["WanAttnProcessor"] = processor_cls

    diffusers.__dict__["models"] = models
    models.__dict__["transformers"] = transformers
    transformers.__dict__["transformer_wan"] = transformer_wan

    sys.modules["diffusers"] = diffusers
    sys.modules["diffusers.models"] = models
    sys.modules["diffusers.models.transformers"] = transformers
    sys.modules["diffusers.models.transformers.transformer_wan"] = transformer_wan
    try:
        yield
    finally:
        for name, module in prior.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_wan_engine_satisfies_engine_protocol() -> None:
    assert isinstance(WanEngine(backend=ROCmBackend()), WorldModelEngine)


def test_wan_engine_info_before_load() -> None:
    engine = WanEngine(backend=ROCmBackend())
    info = engine.info()
    assert info.model_name == "wan-2.2-t2v-a14b"
    assert info.backend == "rocm"
    assert info.dtype == "bfloat16"
    assert info.ready is False  # lazy: no weights touched yet


def test_wan_engine_is_not_loaded_on_construction() -> None:
    assert WanEngine(backend=ROCmBackend()).is_loaded is False


def test_wan_engine_pipeline_raises_before_load() -> None:
    import pytest

    engine = WanEngine(backend=ROCmBackend())
    with pytest.raises(RuntimeError, match="not loaded"):
        _ = engine.pipeline


def test_wan_default_repo_is_t2v_a14b_diffusers() -> None:
    assert WAN_DEFAULT_REPO == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert WanConfig().repo_id == WAN_DEFAULT_REPO


def test_wan_small_repo_is_ti2v_5b() -> None:
    assert WAN_SMALL_REPO == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def test_wan_native_fps_is_sixteen() -> None:
    # Wan-2.2 was trained at 16 FPS; the downstream writer relies on this.
    assert WAN_NATIVE_FPS == 16


def test_wan_config_defaults() -> None:
    config = WanConfig()
    # BF16 is native on CDNA3; FP32 VAE matches the Wan reference path.
    assert config.dtype == "bfloat16"
    assert config.vae_dtype == "float32"
    # Wan-2.2-T2V-A14B model-card default for second-stage guidance.
    assert config.guidance_scale_2 == 3.0
    # Forward-compat hooks (Phase-2 native-loop follow-up).
    assert config.use_native_loop is False
    assert config.cache_skip_every == 0


def test_wan_engine_accepts_custom_config() -> None:
    config = WanConfig(device_index=1, dtype="float16", guidance_scale_2=2.5)
    engine = WanEngine(backend=ROCmBackend(), config=config)
    info = engine.info()
    assert info.dtype == "float16"
    assert info.device == "rocm:1"


def test_wan_profile_exports_at_bench_top_level() -> None:
    # The Wan-specific profiler mirrors profile_cosmos in shape and lives in
    # the same package. Both should re-export from repercep.bench so external
    # callers can pick one without touching the submodule path.
    import repercep.bench as bench

    assert hasattr(bench, "WanProfile")
    assert hasattr(bench, "profile_wan")


def test_wan_profile_schema_includes_moe_split() -> None:
    # The MoE second-expert handoff is non-trivial for Wan-2.2 (Session 10
    # noted ~270 s of the 326 s smoke run was non-DiT work). The profiler must
    # report transformer and transformer_2 timings as separate fields so the
    # handoff is visible in the breakdown — not collapsed into a single
    # dit_loop_s.
    from repercep.bench.profile import WanProfile

    fields = set(WanProfile.model_fields)
    assert "dit_high_noise_s" in fields
    assert "dit_low_noise_s" in fields
    assert "dit_high_noise_calls" in fields
    assert "dit_low_noise_calls" in fields
    # And dit_loop_s = high + low, same units as the Cosmos counterpart so
    # the two reports are directly comparable.
    assert "dit_loop_s" in fields


def test_wan_profile_dit_share_zero_when_total_zero() -> None:
    from repercep.bench.profile import WanProfile

    prof = WanProfile(
        total_s=0.0,
        text_encode_s=0.0,
        dit_loop_s=0.0,
        dit_high_noise_s=0.0,
        dit_low_noise_s=0.0,
        vae_decode_s=0.0,
        other_s=0.0,
        dit_calls=0,
        dit_high_noise_calls=0,
        dit_low_noise_calls=0,
        text_encode_calls=0,
        vae_decode_calls=0,
        compiled=False,
    )
    assert prof.dit_share == 0.0


def test_wan_config_vae_tiling_default_off() -> None:
    # Default-off keeps the MI300X (192 GiB) baseline path unchanged; the flag
    # is opt-in for the 80 GiB H100 envelope.
    assert WanConfig().vae_tiling is False


def test_wan_engine_load_calls_enable_tiling_when_vae_tiling_true() -> None:
    """vae_tiling=True must invoke pipe.vae.enable_tiling() during load()."""
    from unittest.mock import MagicMock

    fake_vae = MagicMock(name="vae")
    fake_pipe = MagicMock(name="pipe")
    fake_pipe.vae = fake_vae

    with _fake_diffusers_loader(fake_vae, fake_pipe):
        engine = WanEngine(backend=ROCmBackend(), config=WanConfig(vae_tiling=True))
        engine.load()

    fake_vae.enable_tiling.assert_called_once()


def test_wan_engine_load_does_not_call_enable_tiling_when_vae_tiling_false() -> None:
    """vae_tiling=False (default) must NOT touch enable_tiling."""
    from unittest.mock import MagicMock

    fake_vae = MagicMock(name="vae")
    fake_pipe = MagicMock(name="pipe")
    fake_pipe.vae = fake_vae

    with _fake_diffusers_loader(fake_vae, fake_pipe):
        engine = WanEngine(backend=ROCmBackend(), config=WanConfig(vae_tiling=False))
        engine.load()

    fake_vae.enable_tiling.assert_not_called()


def test_wan_attention_installer_noops_without_env(monkeypatch: Any) -> None:
    from repercep.attention.wan_processor import maybe_install_repercep_wan_attention

    class FakeTransformer:
        def __init__(self) -> None:
            self.processor: Any | None = None

        def set_attn_processor(self, processor: Any) -> None:
            self.processor = processor

    fake_pipe = types.SimpleNamespace(transformer=FakeTransformer(), transformer_2=None)
    monkeypatch.delenv("REPERCEP_FP8_ATTENTION", raising=False)

    assert maybe_install_repercep_wan_attention(fake_pipe) is False
    assert fake_pipe.transformer.processor is None


def test_wan_attention_installer_sets_repercep_backend(monkeypatch: Any) -> None:
    from repercep.attention import diffusers_backend
    from repercep.attention.wan_processor import maybe_install_repercep_wan_attention

    class FakeWanAttnProcessor:
        _attention_backend: Any
        _repercep_attention_backend: str

        pass

    class FakeTransformer:
        def __init__(self) -> None:
            self.processor: Any | None = None

        def set_attn_processor(self, processor: Any) -> None:
            self.processor = processor

    fake_pipe = types.SimpleNamespace(
        transformer=FakeTransformer(),
        transformer_2=FakeTransformer(),
    )
    monkeypatch.setenv("REPERCEP_FP8_ATTENTION", "fa")
    monkeypatch.setattr(
        diffusers_backend,
        "register_repercep_fp8_backend",
        lambda: "repercep-backend",
    )

    with _fake_wan_processor_module(FakeWanAttnProcessor):
        assert maybe_install_repercep_wan_attention(fake_pipe) is True

    for transformer in (fake_pipe.transformer, fake_pipe.transformer_2):
        assert isinstance(transformer.processor, FakeWanAttnProcessor)
        assert transformer.processor._attention_backend == "repercep-backend"
        assert transformer.processor._repercep_attention_backend == "repercep_fp8"


def test_wan_engine_load_invokes_attention_installer(monkeypatch: Any) -> None:
    from unittest.mock import MagicMock

    import repercep.attention.wan_processor as wan_processor

    fake_vae = MagicMock(name="vae")
    fake_pipe = MagicMock(name="pipe")
    fake_pipe.vae = fake_vae
    calls: list[Any] = []

    def fake_install(pipe: Any) -> bool:
        calls.append(pipe)
        return True

    monkeypatch.setattr(wan_processor, "maybe_install_repercep_wan_attention", fake_install)

    with _fake_diffusers_loader(fake_vae, fake_pipe):
        engine = WanEngine(backend=ROCmBackend())
        engine.load()

    assert calls == [fake_pipe]


def _make_engine_with_fake_pipe(boundary_ratio: float | None) -> tuple[WanEngine, Any]:
    """Construct a WanEngine whose loaded pipe is a MagicMock with a chosen
    boundary_ratio. Returns (engine, fake_pipe) so the test can inspect the
    kwargs passed to fake_pipe.__call__."""
    from unittest.mock import MagicMock

    fake_pipe = MagicMock(name="pipe")
    fake_pipe.config.boundary_ratio = boundary_ratio
    # Mock the __call__ return: output.frames[0] must be a tensor convertible
    # via _as_frame_tensor. Easiest: return uint8 frames-tensor of shape
    # (T, C, H, W) in [0, 1].
    import torch

    fake_frames = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
    fake_pipe.return_value.frames = [fake_frames]

    # CPUBackend, not ROCmBackend: these tests assert pipe-call kwargs, not
    # device placement, and a CUDA-generator construction on a CPU-only box
    # would fail before the assertion ever runs (torch.Generator(device="cuda")
    # needs the ATen CUDA library loaded).
    engine = WanEngine(backend=CPUBackend())
    engine._pipe = fake_pipe  # bypass load()
    return engine, fake_pipe


def test_wan_engine_passes_guidance_scale_2_for_moe_variant() -> None:
    """A14B (boundary_ratio != None) must receive guidance_scale_2."""
    from repercep.runtime.types import GenerationParams, GenerationRequest

    engine, fake_pipe = _make_engine_with_fake_pipe(boundary_ratio=0.875)
    request = GenerationRequest(
        prompt="x", negative_prompt="y",
        params=GenerationParams(num_frames=2, num_inference_steps=2, height=64, width=64,
                                guidance_scale=4.0, fps=16, seed=0),
    )
    _ = list(engine.generate(request))
    _, kwargs = fake_pipe.call_args
    assert "guidance_scale_2" in kwargs
    assert kwargs["guidance_scale_2"] == 3.0  # WanConfig default


def test_wan_engine_omits_guidance_scale_2_for_non_moe_variant() -> None:
    """TI2V-5B (boundary_ratio == None) must NOT receive guidance_scale_2.

    Diffusers raises ``ValueError: guidance_scale_2 is only supported when
    the pipeline's boundary_ratio is not None`` if the kwarg is passed to a
    non-MoE Wan variant. See BUILD_LOG F39.
    """
    from repercep.runtime.types import GenerationParams, GenerationRequest

    engine, fake_pipe = _make_engine_with_fake_pipe(boundary_ratio=None)
    request = GenerationRequest(
        prompt="x", negative_prompt="y",
        params=GenerationParams(num_frames=2, num_inference_steps=2, height=64, width=64,
                                guidance_scale=4.0, fps=16, seed=0),
    )
    _ = list(engine.generate(request))
    _, kwargs = fake_pipe.call_args
    assert "guidance_scale_2" not in kwargs


def test_wan_profile_dit_share_reports_loop_fraction() -> None:
    from repercep.bench.profile import WanProfile

    # 80% of total in the DiT loop (a typical Wan-shaped breakdown — far more
    # DiT-bound than the 17f/8-step smoke, where VAE+postprocess dominate).
    prof = WanProfile(
        total_s=100.0,
        text_encode_s=1.0,
        dit_loop_s=80.0,
        dit_high_noise_s=40.0,
        dit_low_noise_s=40.0,
        vae_decode_s=10.0,
        other_s=9.0,
        dit_calls=80,
        dit_high_noise_calls=40,
        dit_low_noise_calls=40,
        text_encode_calls=2,
        vae_decode_calls=1,
        compiled=False,
    )
    assert prof.dit_share == 0.8
