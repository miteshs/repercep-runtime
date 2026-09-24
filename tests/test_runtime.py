"""Tests for the Runtime types, latent cache, and stub engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from repercep.runtime.engine import WorldModelEngine
from repercep.runtime.latent_cache import LatentCacheError, PagedLatentCache
from repercep.runtime.stub_engine import StubEngine
from repercep.runtime.types import GenerationParams, GenerationRequest


def test_generation_request_requires_nonempty_prompt() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="")


def test_generation_request_defaults_track_cosmos() -> None:
    req = GenerationRequest(prompt="a robot folding laundry")
    assert req.params.num_frames == 121
    assert req.params.num_inference_steps == 35
    assert req.conditioning.kind.value == "none"


def test_generation_params_reject_unknown_field() -> None:
    with pytest.raises(ValidationError):
        GenerationParams(unknown_knob=3)


def test_generation_params_enforce_bounds() -> None:
    with pytest.raises(ValidationError):
        GenerationParams(num_inference_steps=0)


def test_latent_cache_allocate_and_stats() -> None:
    cache = PagedLatentCache(num_pages=4)
    for i in range(4):
        cache.allocate("req-a", frame_index=i, step=0, current_frame=i)
    stats = cache.stats()
    assert stats.allocated == 4
    assert stats.free == 0


def test_latent_cache_evicts_frame_furthest_behind() -> None:
    cache = PagedLatentCache(num_pages=2)
    behind = cache.allocate("r", frame_index=0, step=0, current_frame=10)
    recent = cache.allocate("r", frame_index=9, step=0, current_frame=10)
    cache.unpin(behind)
    cache.unpin(recent)
    # current_frame jumps to 20: the frame-0 page is furthest behind -> victim.
    cache.allocate("r", frame_index=20, step=1, current_frame=20)
    assert cache.stats().allocated == 2
    assert cache.release_request("r") == 2


def test_latent_cache_raises_when_fully_pinned() -> None:
    cache = PagedLatentCache(num_pages=1)
    cache.allocate("r", frame_index=0, step=0, current_frame=0)  # stays pinned
    with pytest.raises(LatentCacheError):
        cache.allocate("r", frame_index=1, step=0, current_frame=1)


def test_stub_engine_satisfies_engine_protocol() -> None:
    assert isinstance(StubEngine(), WorldModelEngine)


def test_stub_engine_yields_requested_frame_count() -> None:
    pytest.importorskip("torch")
    engine = StubEngine()
    req = GenerationRequest(
        prompt="x",
        params=GenerationParams(num_frames=3, height=64, width=64, seed=0),
    )
    frames = list(engine.generate(req))
    assert len(frames) == 3
    assert tuple(frames[0].pixels.shape) == (64, 64, 3)
