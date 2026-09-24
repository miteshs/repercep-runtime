#!/usr/bin/env python3
"""Verify whether REPERCEP_FP8_ATTENTION=fa actually engages on Wan.

Counter-based probe of the diffusers Repercep-bridge entry points. Patches
``_repercep_fp8_attention`` and ``_native_fallback`` in
``repercep.attention.diffusers_backend``, then runs one Wan smoke generation
(17 f / 8 steps) with ``REPERCEP_FP8_ATTENTION=fa`` active.

If the bridge engages:
  * ``repercep_fp8_attention`` calls > 0  (diffusers dispatcher routed to us)
  * ``native_fallback (FA-3 path)`` calls > 0  (we delegated to HopperFlashAttention)
  * ``native_fallback (SDPA path)`` should be 0 unless conditions disqualify

If F40 is real (bridge does NOT engage on Wan):
  * ``repercep_fp8_attention`` calls == 0  (diffusers bypassed the dispatcher)
  * Wan's attention ran through ``F.scaled_dot_product_attention`` directly,
    which on H100 dispatches to cuDNN-flash internally — but never sees our
    FA-3 build.

Usage::

    REPERCEP_FP8_ATTENTION=fa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/python scripts/trace_wan_attention.py
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--small", action="store_true",
                        help="use TI2V-5B (faster trace, same dispatcher path on H100)")
    args = parser.parse_args()

    # Activate bridge first so register_repercep_fp8_backend() runs at import time.
    os.environ.setdefault("REPERCEP_FP8_ATTENTION", "fa")

    from repercep.attention import diffusers_backend as db

    counters: Counter[str] = Counter()

    orig_dispatcher = db._repercep_fp8_attention
    orig_fallback = db._native_fallback

    def counting_dispatcher(*a: Any, **k: Any) -> Any:
        counters["repercep_fp8_attention (dispatcher entry)"] += 1
        return orig_dispatcher(*a, **k)

    def counting_fallback(query: Any, key: Any, value: Any, **k: Any) -> Any:
        # Inspect whether the FA-3 op exists + would be picked. Mirror the
        # guard inside _native_fallback so the counter labels the same branch
        # the production code takes.
        fa_op = getattr(orig_fallback, "_fa_op", "uninit")
        import torch
        on_cuda = query.device.type == "cuda"
        bf_or_fp16 = query.dtype in (torch.bfloat16, torch.float16)
        no_mask = k.get("attn_mask") is None
        no_dropout = float(k.get("dropout_p", 0.0) or 0.0) == 0.0
        eligible = on_cuda and bf_or_fp16 and no_mask and no_dropout
        if eligible and fa_op not in (None, "uninit"):
            counters["native_fallback -> FA-3 (HopperFlashAttention)"] += 1
        else:
            counters["native_fallback -> SDPA (fallback)"] += 1
        return orig_fallback(query, key, value, **k)

    db._repercep_fp8_attention = counting_dispatcher  # type: ignore[assignment]
    db._native_fallback = counting_fallback  # type: ignore[assignment]

    # Also re-register the patched dispatcher with the diffusers backend
    # registry so the routing target is our wrapper, not the original.
    from diffusers.models.attention_dispatch import (
        _AttentionBackendRegistry,  # type: ignore[attr-defined]
    )
    member = db._ensure_backend_name_registered()
    _AttentionBackendRegistry._backends[member] = counting_dispatcher

    # ALSO patch torch SDPA so we can count calls that bypass the dispatcher
    # entirely (the F40 hypothesis: WanTransformer3DModel calls SDPA directly).
    import torch.nn.functional as torchF  # noqa: N812
    orig_sdpa = torchF.scaled_dot_product_attention

    def counting_sdpa(*a: Any, **k: Any) -> Any:
        counters["torch.F.scaled_dot_product_attention (direct)"] += 1
        return orig_sdpa(*a, **k)

    torchF.scaled_dot_product_attention = counting_sdpa  # type: ignore[assignment]

    # Now run the Wan smoke. Everything downstream sees the patched fns.
    from repercep.backend.registry import select_backend
    from repercep.models.wan import SMALL_REPO, WanConfig, WanEngine
    from repercep.runtime.types import GenerationParams, GenerationRequest

    backend = select_backend()
    config = WanConfig(vae_tiling=True)
    if args.small:
        config.repo_id = SMALL_REPO
    engine = WanEngine(backend, config)
    print(f"[trace] backend={backend.name}  repo={config.repo_id}", flush=True)
    print(f"[trace] REPERCEP_FP8_ATTENTION={os.environ.get('REPERCEP_FP8_ATTENTION')}", flush=True)
    print("[trace] loading model ...", flush=True)
    engine.load()

    request = GenerationRequest(
        prompt=(
            "A sleek autonomous delivery robot rolls along a sunlit city "
            "sidewalk, photorealistic."
        ),
        negative_prompt="blurry, low quality",
        params=GenerationParams(
            num_frames=args.frames,
            num_inference_steps=args.steps,
            height=720,
            width=1280,
            guidance_scale=4.0,
            fps=16,
            seed=0,
        ),
    )
    print(f"[trace] generating {args.frames}f / {args.steps} steps ...", flush=True)
    frames = list(engine.generate(request))
    print(f"[trace] done, {len(frames)} frames", flush=True)

    print()
    print("=== ATTENTION DISPATCHER COUNTERS ===")
    if not counters:
        print("  (no attention calls observed — instrumentation likely missed the path)")
        return 1
    width = max(len(k) for k in counters) + 2
    for label, count in counters.most_common():
        print(f"  {label.ljust(width)} {count}")
    print()
    bridge_calls = counters["repercep_fp8_attention (dispatcher entry)"]
    direct_sdpa = counters["torch.F.scaled_dot_product_attention (direct)"]
    fa3_via_bridge = counters["native_fallback -> FA-3 (HopperFlashAttention)"]
    sdpa_via_bridge = counters["native_fallback -> SDPA (fallback)"]
    print("=== VERDICT ===")
    if bridge_calls == 0 and direct_sdpa > 0:
        print(f"  F40 CONFIRMED: bridge engaged 0x; {direct_sdpa} direct SDPA calls.")
        print("  WanTransformer3DModel bypasses _AttentionBackendRegistry.")
    elif bridge_calls > 0 and fa3_via_bridge > 0:
        print(
            f"  F40 REFUTED: bridge engaged {bridge_calls}x; "
            f"FA-3 took {fa3_via_bridge}/{bridge_calls}.",
        )
    elif bridge_calls > 0 and fa3_via_bridge == 0:
        print(
            f"  PARTIAL: bridge engaged {bridge_calls}x but never took FA-3 "
            f"(SDPA fallback {sdpa_via_bridge}x). Eligibility guard rejected.",
        )
    else:
        print("  Unexpected -- see counts above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
