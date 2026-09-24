"""Tests for the benchmark harness, exercised against the StubEngine."""

from __future__ import annotations

import pytest

from repercep.bench.harness import BenchmarkResult, benchmark_engine, speedup
from repercep.runtime.stub_engine import StubEngine
from repercep.runtime.types import GenerationParams, GenerationRequest


def _small_request() -> GenerationRequest:
    return GenerationRequest(
        prompt="a test clip",
        params=GenerationParams(num_frames=4, height=64, width=64, seed=0),
    )


def test_benchmark_rejects_zero_iters() -> None:
    with pytest.raises(ValueError, match="iters"):
        benchmark_engine(StubEngine(), _small_request(), iters=0)


def test_benchmark_stub_engine() -> None:
    pytest.importorskip("torch")
    result = benchmark_engine(StubEngine(), _small_request(), warmup=1, iters=2)
    assert isinstance(result, BenchmarkResult)
    assert result.num_frames == 4
    assert result.iters == 2
    assert result.mean_latency_s > 0
    assert result.frames_per_second > 0


def test_benchmark_result_serializes_to_json() -> None:
    pytest.importorskip("torch")
    result = benchmark_engine(StubEngine(), _small_request(), iters=1)
    assert '"engine"' in result.model_dump_json()


def test_speedup_ratio() -> None:
    pytest.importorskip("torch")
    request = _small_request()
    a = benchmark_engine(StubEngine(), request, iters=1)
    b = benchmark_engine(StubEngine(), request, iters=1)
    assert speedup(a, b) > 0
