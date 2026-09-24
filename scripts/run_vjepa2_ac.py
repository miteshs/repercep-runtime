"""V-JEPA 2-AC interactive rollout + energy-MPC planning on the Repercep backend.

V-JEPA 2-AC is the action-conditioned world model (ADR-0008): it predicts the
next *state embedding* given an action and plans by minimizing a latent-space
energy toward a goal. This is the closed-loop / energy-based regime, not the
one-shot text->video path.

Two modes:

  * ``--stub`` injects a synthetic linear world model (next = last + action) so
    the rollout, the energy function, and the CEM planner run end-to-end on CPU
    **without any weights** — this is the wiring/CI demonstration.
  * without ``--stub`` it builds the real engine; that needs the V-JEPA 2-AC
    checkpoints + a GPU and will raise ``NotImplementedError`` until the weight
    loaders in ``repercep.models.vjepa2_ac`` are filled in (the docstrings there
    are the spec).

Usage::

    .venv/bin/python scripts/run_vjepa2_ac.py --stub --plan --steps 4
    .venv/bin/python scripts/run_vjepa2_ac.py --backend cuda   # real, needs weights
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _setup() -> None:
    """Make src/ importable when running from a worktree."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))


_setup()


import torch  # noqa: E402

from repercep.backend.registry import select_backend  # noqa: E402
from repercep.models.vjepa2_ac import VJepa2ACConfig, VJepa2ACEngine  # noqa: E402
from repercep.runtime.types import Action, ConditioningInput, RolloutParams  # noqa: E402

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


class _StubEncoder:
    """Returns a fixed ``(1, T, D)`` feature tensor — stands in for V-JEPA 2."""

    def __init__(self, context: torch.Tensor) -> None:
        self._context = context

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self._context.unsqueeze(0)


class _StubPredictor:
    """Synthetic linear dynamics: next state = last context frame + action."""

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return context[-1] + action


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="fp32")
    p.add_argument("--steps", type=int, default=4, help="open-loop rollout steps")
    p.add_argument("--horizon", type=int, default=4, help="planning horizon")
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--plan", action="store_true", help="run energy-MPC toward a random goal")
    p.add_argument("--stub", action="store_true", help="synthetic world model (no weights)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    torch.manual_seed(args.seed)
    # The real V-JEPA 2-AC is a 7-DoF action-conditioned model; the stub may use
    # any control dim.
    action_dim = args.action_dim if args.stub else 7
    cfg = VJepa2ACConfig(action_dim=action_dim, context_frames=8, dtype=_DTYPES[args.dtype])

    if args.stub:
        ctx0 = torch.zeros(2, cfg.action_dim)
        engine = VJepa2ACEngine(
            backend, cfg, encoder=_StubEncoder(ctx0), predictor=_StubPredictor()
        )
    else:
        engine = VJepa2ACEngine(backend, cfg)
        print("[repercep] no --stub: this needs the V-JEPA 2-AC weights (ADR-0008).", flush=True)

    print(f"[repercep] backend={backend.name} dtype={cfg.dtype} action_dim={cfg.action_dim}")
    state = engine.reset(ConditioningInput(), RolloutParams(horizon=args.horizon))
    print(f"[repercep] reset: step={state.step_index} context={tuple(state.context.shape)}")

    for _ in range(args.steps):
        action = Action(values=torch.randn(cfg.action_dim).tolist())
        state, step = engine.step(state, action)
        print(f"[repercep] step {step.step_index}: context={tuple(state.context.shape)}")

    result = {
        "backend": backend.name,
        "steps": args.steps,
        "final_step": state.step_index,
        "context_shape": list(state.context.shape),
    }

    if args.plan:
        # Build a *reachable* goal: the frame one known action takes us to, then
        # plan back toward it. The goal is frame-shaped (tokens_per_frame x D),
        # matching what `_rollout_energy` compares against — for both the stub
        # (tokens_per_frame=1) and the real model (tokens_per_frame=P).
        tpf = engine._tokens_per_frame
        goal_action = Action(values=torch.randn(cfg.action_dim).tolist())
        goal_state, _ = engine.step(state, goal_action)
        goal = goal_state.context[-tpf:]
        zeros = torch.zeros(args.horizon, cfg.action_dim)
        e_zero = float(engine._rollout_energy(state, zeros, goal))
        sequence = engine._plan_sequence(state, goal, horizon=args.horizon)
        e_planned = float(engine._rollout_energy(state, sequence, goal))
        action = engine.plan(state, goal, horizon=args.horizon)
        n_act = len(action.values)
        print(f"[repercep] plan: energy {e_zero:.3f} -> {e_planned:.3f} (first action dim={n_act})")
        result["energy_zero"] = round(e_zero, 4)
        result["energy_planned"] = round(e_planned, 4)

    print(f"[repercep] RESULT {json.dumps(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
