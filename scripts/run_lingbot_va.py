#!/usr/bin/env python3
"""Drive the REAL LingBot-VA 2.0 through the Repercep interactive seam.

The Phase-1 GPU verify: imagination-mode rollout via
``LingBotVAEngine.reset → plan → step`` (one ``infer_chunk`` per step, the
model's own proposed actions executed), timed per chunk so the result is
directly comparable to the reference-stack numbers in
``docs/LINGBOT_VA_ON_H100.md`` (warm mean 1384.7 ms there). Emits a verbatim
``RESULT`` JSON line.

    python scripts/run_lingbot_va.py \
        --obs-dir /workspace/lingbot-va/example/demo \
        --prompt "Pick the green cube and place it inside the blue box"

Needs the research repo importable as ``wan_va`` and the checkpoint bundle
(see docs/LINGBOT_VA_PORT_PLAN.md §4).
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
from repercep.models.lingbot_va import LingBotVAConfig, LingBotVAEngine  # noqa: E402
from repercep.runtime.types import (  # noqa: E402
    Action,
    ConditioningInput,
    ConditioningKind,
    RolloutParams,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="robbyant/lingbot-va-base", help="HF id or local bundle dir")
    ap.add_argument("--obs-dir", required=True, help="dir with <cam_key>.png seed images")
    ap.add_argument("--prompt", default="Pick the green cube and place it inside the blue box")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm"], default="auto")
    ap.add_argument("--save-latents", default=None, help="optional .pt path for the rollout")
    args = ap.parse_args()

    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    engine = LingBotVAEngine(backend, LingBotVAConfig(repo=args.repo, prompt=args.prompt))

    t = time.perf_counter()
    engine.load()
    _sync()
    load_s = time.perf_counter() - t
    print(f"[seam] load {load_s:.1f}s ready={engine.info().ready}", flush=True)

    t = time.perf_counter()
    state = engine.reset(
        ConditioningInput(kind=ConditioningKind.IMAGE, uri=args.obs_dir), RolloutParams()
    )
    _sync()
    reset_s = time.perf_counter() - t
    print(f"[seam] reset {reset_s:.2f}s context={tuple(state.context.shape)}", flush=True)

    chunk_ms: list[float] = []
    latents = [state.context.cpu()]
    goal = torch.zeros(1)  # policy-regime: the prompt is the goal; tensor unused
    for i in range(args.chunks):
        _sync()
        t = time.perf_counter()
        action = engine.plan(state, goal, horizon=engine._config.frame_chunk_size)
        proposed = engine._sessions[state.session_id].pending_actions
        assert proposed is not None
        state, step = engine.step(state, Action(values=proposed.flatten().tolist()))
        _sync()
        dt_ms = (time.perf_counter() - t) * 1000
        chunk_ms.append(round(dt_ms, 1))
        print(
            f"[seam] chunk {i}: {dt_ms:.0f} ms  step_index={step.step_index} "
            f"a0[:3]={[round(v, 2) for v in action.values[:3]]}",
            flush=True,
        )
        latents.append(state.context.cpu())

    if args.save_latents:
        torch.save(torch.stack(latents[1:]), args.save_latents)

    warm = chunk_ms[2:] if len(chunk_ms) > 2 else chunk_ms
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    result = {
        "bench": "lingbot_va_seam",
        "model": args.repo,
        "device": backend.devices()[0].name,
        "load_s": round(load_s, 1),
        "reset_s": round(reset_s, 2),
        "chunk_ms": chunk_ms,
        "chunk_ms_warm_mean": round(sum(warm) / len(warm), 1),
        "reference_stack_warm_ms": 1384.7,
        "peak_hbm_gib": round(peak, 1),
        "context_shape": list(state.context.shape),
    }
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
