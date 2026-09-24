"""Benchmark harness — measurement before optimization.

The implementation plan is explicit: build the benchmark suite *before* the
optimizer, so every later speedup is measured against a fixed baseline. This
harness times any ``WorldModelEngine`` and reports latency, throughput, and
peak HBM.

Today the Cosmos engine runs the stock ``diffusers`` pipeline, so benchmarking
it establishes exactly the *naive PyTorch + Diffusers baseline* the plan calls
for. When Repercep-specific optimizations land, ``speedup()`` compares a
candidate run against that baseline.
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from repercep.runtime.engine import WorldModelEngine
    from repercep.runtime.types import GenerationRequest


class BenchmarkResult(BaseModel):
    """Timing and resource summary of a benchmarked run.

    A Pydantic model so results serialize straight to JSON for the future
    Studio regression dashboards.
    """

    model_config = ConfigDict(extra="forbid")

    engine: str
    device: str
    num_frames: int
    num_inference_steps: int
    iters: int
    mean_latency_s: float
    p50_latency_s: float
    min_latency_s: float
    frames_per_second: float
    peak_vram_gib: float


def benchmark_engine(
    engine: WorldModelEngine,
    request: GenerationRequest,
    *,
    warmup: int = 0,
    iters: int = 1,
) -> BenchmarkResult:
    """Time ``engine.generate(request)`` over ``iters`` runs.

    Args:
        engine: the world-model engine under test.
        request: the generation request to replay each iteration.
        warmup: untimed runs first (kernel autotune, allocator warmup).
        iters: timed runs; latency statistics are computed across them.

    Raises:
        ValueError: ``iters`` is less than 1.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")

    import torch

    on_gpu = torch.cuda.is_available()

    # If the engine loads weights lazily, do it now so the benchmark measures
    # generation, not the one-time model load.
    loader = getattr(engine, "load", None)
    if callable(loader):
        loader()

    for _ in range(warmup):
        _drain(engine, request)
    if on_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    frames = 0
    for _ in range(iters):
        start = time.perf_counter()
        frames = _drain(engine, request)
        if on_gpu:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    mean = statistics.mean(latencies)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if on_gpu else 0.0
    info = engine.info()
    return BenchmarkResult(
        engine=info.model_name,
        device=info.device,
        num_frames=frames,
        num_inference_steps=request.params.num_inference_steps,
        iters=iters,
        mean_latency_s=mean,
        p50_latency_s=statistics.median(latencies),
        min_latency_s=min(latencies),
        frames_per_second=frames / mean if mean > 0 else 0.0,
        peak_vram_gib=peak_vram,
    )


def speedup(baseline: BenchmarkResult, candidate: BenchmarkResult) -> float:
    """Latency speedup of ``candidate`` over ``baseline`` (>1.0 means faster)."""
    return baseline.mean_latency_s / candidate.mean_latency_s


def _drain(engine: WorldModelEngine, request: GenerationRequest) -> int:
    """Run one full generation, returning the number of frames produced."""
    return sum(1 for _ in engine.generate(request))
