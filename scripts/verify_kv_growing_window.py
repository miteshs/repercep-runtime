#!/usr/bin/env python3
"""GPU-verify the real AC-predictor KV-cache adapter (ADR-0009 Finding 3).

``tests/test_interactive.py::test_ac_predictor_adapter_cached_growth_matches_full_forward``
proves this algorithm correct against a miniature *synthetic* predictor that
shares the real model's augmented-token/RoPE/block-causal structure. This
script runs the identical growth loop (``init_cache`` -> ``step_cached`` ->
``append_frame``, compared each step against the uncached full forward)
against the REAL pretrained V-JEPA 2-AC predictor, so the integration itself
-- not just the algorithm shape -- is checked on real weights, not assumed
from the CPU parity test.

    python scripts/verify_kv_growing_window.py                # H100/ROCm box
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _setup() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))


_setup()

import torch  # noqa: E402

from repercep.backend.registry import select_backend  # noqa: E402
from repercep.models.vjepa2_ac import VJepa2ACConfig, VJepa2ACEngine  # noqa: E402
from repercep.runtime.types import ConditioningInput, RolloutParams  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
    ap.add_argument("--context-frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="encoder context/action/output dtype -- float32 isolates the cached-"
        "vs-uncached algorithm from bf16 boundary rounding (the ADR's original "
        "'3.8e-4, within fp32 tolerance' claim was measured at fp32)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    dev = backend.devices()[0]
    print(f"[kv-verify] backend={backend.name} device={dev.name}", flush=True)

    # reset_context_frames=1 starts the session below context_frames, so every
    # step in this loop is a pure-growth step -- the regime this ADR's design
    # claims is exact (no eviction, which Finding 2 showed is NOT exact).
    cfg = VJepa2ACConfig(
        context_frames=args.context_frames, reset_context_frames=1, dtype=args.dtype
    )
    engine = VJepa2ACEngine(backend, cfg)
    t = time.perf_counter()
    engine.load()
    load_s = round(time.perf_counter() - t, 1)
    print(f"[kv-verify] load: {load_s}s", flush=True)

    state = engine.reset(ConditioningInput(), RolloutParams(horizon=args.context_frames))
    adapter = engine._predictor
    assert adapter is not None and getattr(adapter, "supports_kv_cache", False)
    action_dim = engine._config.action_dim

    context = state.context
    cache = adapter.init_cache(context)
    diffs: list[float] = []
    for i in range(args.steps):
        action = torch.randn(action_dim, device=context.device, dtype=context.dtype)
        expected = adapter(context, action)
        actual, staged = adapter.step_cached(cache, action)
        abs_diff = (actual.float() - expected.float()).abs()
        diff = abs_diff.max().item()
        rel = (abs_diff / expected.float().abs().clamp_min(1e-6)).max().item()
        diffs.append(diff)
        print(
            f"[kv-verify] step {i}: max abs diff = {diff:.6e}, max rel diff = {rel:.6e}, "
            f"expected |max|={expected.float().abs().max().item():.4f}",
            flush=True,
        )
        cache = adapter.append_frame(staged, actual)
        context = torch.cat([context, actual], dim=0)

    result = {
        "verify": "kv_growing_window_gpu",
        "device": dev.name,
        "context_frames_cap": args.context_frames,
        "steps": args.steps,
        "max_abs_diff": max(diffs),
        "diffs": [round(d, 8) for d in diffs],
        "load_seconds": load_s,
    }
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
