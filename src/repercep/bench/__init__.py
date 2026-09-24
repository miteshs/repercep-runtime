"""Benchmark harness for the Repercep Runtime.

Measurement before optimization: this package times any ``WorldModelEngine``
and reports latency, throughput, and peak HBM, and profiles the Cosmos pipeline
stage by stage — so every later speedup is measured against a fixed baseline.
"""

from __future__ import annotations

from repercep.bench.harness import BenchmarkResult, benchmark_engine, speedup
from repercep.bench.profile import CosmosProfile, WanProfile, profile_cosmos, profile_wan

__all__ = [
    "BenchmarkResult",
    "CosmosProfile",
    "WanProfile",
    "benchmark_engine",
    "profile_cosmos",
    "profile_wan",
    "speedup",
]
