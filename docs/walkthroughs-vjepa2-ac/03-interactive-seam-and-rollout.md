# Part 3 — The interactive seam and the latent rollout

Part 2 gave us `get_vision_features(video) → (B, N, D)`. This part is the seam
that turns that into a *rolling* world state you step with actions:
`src/repercep/runtime/interactive.py` (the Protocol) and `src/repercep/models/
vjepa2_ac.py` (`reset` / `step`). This is Repercep's own code.

## 3.1 A different seam than Cosmos

The Cosmos path uses `WorldModelEngine.generate(request) -> Iterator[Frame]` —
one request, one stream, stateless. That cannot express a *loop* where the client
injects an action each step. So Repercep adds a second Protocol,
`InteractiveWorldModel`:

```python
@runtime_checkable
class InteractiveWorldModel(Protocol):
    def info(self) -> EngineInfo: ...
    def reset(self, conditioning, params) -> WorldState: ...
    def step(self, state, action) -> tuple[WorldState, LatentStep]: ...
    def plan(self, state, goal, horizon) -> Action: ...
```

`reset → step(action) → step(action) → …`, with `plan` (Part 4) as the
energy-minimizing move. It reuses the *same* `Backend` seam (ADR-0003) and
`EngineInfo` contract as Cosmos — only the **shape of interaction** differs.

## 3.2 The state that flows (runtime/types.py)

- `WorldState` — in-process, like `Frame`: it carries a live tensor and never
  crosses the wire. `context` = the rolling window of state embeddings
  `(T_ctx, D)`, plus `step_index`, `session_id`.
- `Action` (wire) — `values: list[float]` + a `space` tag. A control vector, not
  pixels.
- `LatentStep` (wire envelope, like `FrameChunk`) — `step_index`, optional
  `energy`, optional `frame` (only if a pixel decoder is attached — the AVID
  sibling, Part 5). No tensor crosses the wire.
- `ResetRequest` / `RolloutParams` — reuse Cosmos's `ConditioningInput`
  (image/video URI) to seed, plus a `horizon`.

## 3.3 `reset` — observation → initial world state

```python
def reset(self, conditioning, params):
    self._ensure_encoder()                          # HF V-JEPA 2 encoder (Part 2; the port)
    frames = self._resolve_frames(conditioning)     # NONE -> synthetic seed clip; URI decode = Phase 2
    with torch.inference_mode():
        features = self._encoder.get_vision_features(pixel_values_videos=frames)   # (B, N, D)  (Part 2)
    context = _as_context(features)[-self._config.context_frames:]                 # -> (T_ctx, D)
    return WorldState(context=context, step_index=0, session_id=_new_session_id())
```

`_as_context` squeezes the batch axis to `(N, D)`; the last `context_frames` are
kept as the block-causal window. `inference_mode` for the same reason as the
Cosmos loop (the F18 memory fix).

## 3.4 `step` — one action, one next latent

```python
def step(self, state, action):
    self._ensure_predictor()                        # the AC head (Part 2; the port)
    with torch.inference_mode():
        vec = torch.tensor(action.values, dtype=state.context.dtype, device=state.context.device)
        nxt = self._predictor(state.context, vec)               # (context, action) -> next embedding
        context = torch.cat([state.context, nxt.unsqueeze(0)], dim=0)[-self._config.context_frames:]
    new = WorldState(context=context, step_index=state.step_index + 1, session_id=state.session_id)
    return new, LatentStep(step_index=new.step_index)
```

Two things to notice:

- **The block-causal sliding window.** Each step appends the predicted next-state
  embedding and keeps the last `context_frames` — the bounded recent history the
  AC predictor attends over (Part 2's block-causal head). Bounded window ⇒
  bounded per-step cost and memory, no matter how long the rollout. (Contrast
  Cosmos, where every step reprocesses the *whole* S ≈ 109k latent volume.)
- **It does not mutate `state`.** It returns a *new* `WorldState`. Deliberate: the
  planner (Part 4) branches many candidate rollouts from a shared prefix, so
  `step` must be side-effect-free on its input.

## 3.5 The injectable predictor — why this runs without weights

`step` calls `self._predictor(context, action)`, where `_predictor` satisfies a
tiny `_Predictor` Protocol (`__call__(context, action) -> tensor`). In production
it's the ported AC head; in tests and `--stub` it's an injected toy
(`next = last + action`). That injection is what lets the **entire rollout +
planner** run and be unit-tested on CPU with no weights.

## Run it (CPU)

`scripts/run_vjepa2_ac.py --stub` exercises exactly this — the context window
grows, then caps at `context_frames`:

```
[repercep] reset: step=0 context=(2, 4)
[repercep] step 1: context=(3, 4)
[repercep] step 2: context=(4, 4)
[repercep] step 3: context=(5, 4)
```

and `tests/test_interactive.py` (`test_reset_and_step_advance_context`,
`test_context_window_is_capped`) asserts the shapes + the window cap on CPU.

**Next:** Part 4 — `plan`: searching action sequences to minimize the latent
energy.
