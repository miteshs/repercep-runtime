# Part 4 — Energy-based planning

This is the payoff — where the world model is *used*. Given a goal embedding,
find the actions that drive the rollout to low energy against it. It's
`models/vjepa2_ac.py` `plan` / `_plan_sequence` / `_rollout_energy` — the concrete
instance of Part 1 §1.5 (planning = energy minimization).

## 4.1 The energy of a plan

```python
def _rollout_energy(self, state, action_seq, goal):
    rolled = state
    for t in range(int(action_seq.shape[0])):
        rolled, _ = self.step(rolled, Action(values=action_seq[t].tolist()))   # roll forward (Part 3)
    energy = torch.linalg.vector_norm(rolled.context[-1] - goal)               # ||s_T - goal||
    return energy
```

Roll the world model forward under a candidate action sequence (via `step`,
Part 3), then measure how far the terminal latent lands from the goal embedding.
**That distance is the EBM energy** (Part 1): low = this plan reaches the goal.
Because `step` doesn't mutate, every candidate branches cleanly from the same
`state`.

## 4.2 The search — cross-entropy method (CEM)

```python
def _plan_sequence(self, state, goal, horizon):
    mean = torch.zeros(horizon, action_dim); std = torch.ones_like(mean)
    for _ in range(plan_iters):
        seqs = mean + std * torch.randn(plan_samples, horizon, action_dim)        # sample candidates
        energies = torch.stack([self._rollout_energy(state, seqs[i], goal)
                                for i in range(plan_samples)])
        elite = seqs[energies.topk(plan_elites, largest=False).indices]           # keep lowest-energy
        mean, std = elite.mean(0), elite.std(0).clamp_min(1e-6)                   # refit to the elites
    return mean                                                                   # elite-mean sequence

def plan(self, state, goal, horizon):
    return Action(values=self._plan_sequence(state, goal, horizon)[0].tolist(), space="ee_delta")
```

CEM is the simplest sampling-based optimizer: sample action sequences from a
Gaussian, score each by `_rollout_energy`, keep the **elite** (lowest-energy)
ones, refit the Gaussian to them, repeat. The mean concentrates on low-energy
plans; `plan` returns the **first action** of that mean sequence — you re-plan
each real step (model-predictive control). **This is energy minimization over
action sequences**, exactly Part 1 §1.5.

## 4.3 The verified demo, dissected

`scripts/run_vjepa2_ac.py --stub --plan` (Part 0) printed:

```
[repercep] plan: energy 1.763 -> 0.401 (first action dim=4)
```

With the stub dynamics `next = last + action`, after H steps the terminal latent
is `s₀ + Σ aₜ`, so `energy = ‖(s₀ + Σ aₜ) − goal‖` is minimized when `Σ aₜ` equals
the goal offset. CEM finds action sequences whose sum points at the goal, driving
the energy from **1.763** (the zero-action baseline `‖goal − s₀‖`) down to
**0.401** (~4.4× closer). `tests/test_interactive.py::test_plan_reduces_energy_
toward_goal` asserts exactly that — `energy(planned) < energy(zero-action)` — on
CPU.

## 4.4 Why this is cheap, and where it goes (systems)

Cost per planning call = `plan_samples × horizon` predictor forwards (defaults
64 × small). Each is a **small block-causal head over a bounded latent window**
(Part 3) — cheap, because it's embeddings, not pixels (Part 1 §1.6). The search is
**embarrassingly parallel** over the `plan_samples` candidates; today the rollout
loop is sequential — batching the candidates is a clear throughput win left for
the port. Contrast: doing this with a *diffusion* video model means a full
multi-second generation **per candidate** — thousands of DiT forwards per planning
step. *That* is why the latent/energy path, not the diffusion path, is the one
suited to closed-loop control.

With the real AC predictor (the Part 2 port) and a real `goal` — e.g. the encoder
embedding of a goal image — this is robot/agent planning: "what actions get me to
that state?" The algorithm here is real and tested; its *quality* rides on the
predictor weights.

## Run it (CPU)

```bash
python scripts/run_vjepa2_ac.py --stub --plan --steps 4 --horizon 4
python -m pytest tests/test_interactive.py -q -k plan      # the energy-reduction test
```

**Next:** Part 5 — serving this loop over a WebSocket, and what the real thing
needs (weights, goal embeddings, the AVID pixel sibling).
