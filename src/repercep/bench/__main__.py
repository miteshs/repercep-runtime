"""``python -m repercep.bench`` — run a benchmark from the command line.

Examples:
    # quick sanity benchmark, no model weights needed
    python -m repercep.bench --engine stub

    # the real Cosmos-Predict-7B baseline on the MI300X
    python -m repercep.bench --engine cosmos --frames 121 --steps 36 --warmup 1 --iters 3
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from repercep.runtime.engine import WorldModelEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="repercep.bench", description="Benchmark a Repercep world-model engine."
    )
    parser.add_argument("--engine", choices=("stub", "cosmos"), default="stub")
    parser.add_argument("--prompt", default="a quiet street at dawn, first-person view")
    parser.add_argument("--frames", type=int, default=None, help="override num_frames")
    parser.add_argument("--steps", type=int, default=None, help="override steps")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args(argv)

    from repercep.bench.harness import benchmark_engine
    from repercep.runtime.types import GenerationParams, GenerationRequest

    overrides: dict[str, int] = {"seed": args.seed}
    if args.frames is not None:
        overrides["num_frames"] = args.frames
    if args.steps is not None:
        overrides["num_inference_steps"] = args.steps
    if args.height is not None:
        overrides["height"] = args.height
    if args.width is not None:
        overrides["width"] = args.width

    request = GenerationRequest(prompt=args.prompt, params=GenerationParams(**overrides))
    engine = _build_engine(args.engine)
    result = benchmark_engine(engine, request, warmup=args.warmup, iters=args.iters)
    print(result.model_dump_json(indent=2))
    return 0


def _build_engine(kind: str) -> WorldModelEngine:
    if kind == "stub":
        from repercep.runtime.stub_engine import StubEngine

        return StubEngine()
    from repercep.backend.registry import select_backend
    from repercep.models.cosmos import CosmosEngine

    return CosmosEngine(backend=select_backend())


if __name__ == "__main__":
    raise SystemExit(main())
