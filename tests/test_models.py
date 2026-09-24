"""Tests for the Cosmos engine.

Construction, identity, and Protocol conformance are checked here. Actual
generation needs ~38 GB of downloaded weights and a GPU, and is exercised by
the run/benchmark scripts rather than the unit suite.
"""

from __future__ import annotations

from repercep.backend.rocm import ROCmBackend
from repercep.models.cosmos import DEFAULT_REPO, CosmosConfig, CosmosEngine
from repercep.runtime.engine import WorldModelEngine


def test_cosmos_engine_satisfies_engine_protocol() -> None:
    assert isinstance(CosmosEngine(backend=ROCmBackend()), WorldModelEngine)


def test_cosmos_engine_info_before_load() -> None:
    engine = CosmosEngine(backend=ROCmBackend())
    info = engine.info()
    assert info.model_name == "cosmos-predict1-7b-text2world"
    assert info.backend == "rocm"
    assert info.ready is False  # lazy: no weights touched yet


def test_cosmos_engine_is_not_loaded_on_construction() -> None:
    assert CosmosEngine(backend=ROCmBackend()).is_loaded is False


def test_default_repo_is_the_diffusers_format() -> None:
    assert DEFAULT_REPO == "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
    assert CosmosConfig().repo_id == DEFAULT_REPO


def test_cosmos_config_caching_defaults() -> None:
    """The caching defaults document the BUILD_LOG-recorded headline behaviour:
    no caching out of the box; adaptive gates documented but not the default.
    """
    cfg = CosmosConfig()
    assert cfg.cache_skip_every == 0
    assert cfg.cache_warmup_steps == 4
    assert cfg.cache_mode == "none"
    # Adaptive tuned default on 121f/36 step (see scripts/bench_caching.py).
    assert cfg.cache_adaptive_threshold == 0.3
    assert cfg.cache_force_full_every == 16


def test_cosmos_config_compile_gate_defaults() -> None:
    """The F15/F18 frame-size compile gate defaults must trip somewhere
    between the known-safe 49f and the known-broken 121f shape.
    """
    cfg = CosmosConfig()
    assert cfg.compile_transformer is False  # opt-in
    assert cfg.compile_mode is None  # inductor default
    assert cfg.compile_dynamic is False
    assert 49 < cfg.compile_max_frames < 121


def test_cosmos_engine_gate_is_noop_when_uncompiled() -> None:
    """If ``compile_transformer`` is False, the gate has nothing to swap and
    must do nothing for any ``num_frames`` value.
    """
    engine = CosmosEngine(
        backend=ROCmBackend(),
        config=CosmosConfig(compile_transformer=False),
    )
    # No load() here — the gate must tolerate the unloaded state too.
    engine._gate_compile_if_needed(121)
    assert engine._compile_gated is False


def test_cosmos_engine_gate_swaps_when_threshold_exceeded() -> None:
    """Simulate a loaded compiled pipe; the gate must swap the transformer
    for ``num_frames`` > ``compile_max_frames`` and leave it alone below.
    """

    class _Original:
        pass

    class _Compiled:
        pass

    class _Pipe:
        def __init__(self) -> None:
            self.transformer: object = _Compiled()

    engine = CosmosEngine(
        backend=ROCmBackend(),
        config=CosmosConfig(compile_transformer=True, compile_max_frames=64),
    )
    # Manually wire the bookkeeping that ``load()`` does on a real run.
    original = _Original()
    pipe = _Pipe()
    engine._pipe = pipe
    engine._uncompiled_transformer = original

    # Under the ceiling: no swap, gate not tripped.
    engine._gate_compile_if_needed(49)
    assert pipe.transformer is not original
    assert engine._compile_gated is False

    # Over the ceiling: swap, gate tripped. Then re-running is a no-op.
    engine._gate_compile_if_needed(121)
    engine._gate_compile_if_needed(121)
    assert pipe.transformer is original
    assert engine._compile_gated is True


def test_cosmos_engine_gate_disabled_when_max_frames_zero() -> None:
    """``compile_max_frames=0`` opts out of the gate entirely (caller knows
    upstream is fixed). The compiled transformer must remain in use.
    """

    class _Original:
        pass

    class _Compiled:
        pass

    class _Pipe:
        def __init__(self) -> None:
            self.transformer: object = _Compiled()

    engine = CosmosEngine(
        backend=ROCmBackend(),
        config=CosmosConfig(compile_transformer=True, compile_max_frames=0),
    )
    original = _Original()
    pipe = _Pipe()
    engine._pipe = pipe
    engine._uncompiled_transformer = original

    engine._gate_compile_if_needed(121)
    assert pipe.transformer is not original  # not swapped
    assert engine._compile_gated is False
