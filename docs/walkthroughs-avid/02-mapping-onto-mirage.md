# Part 2 — Mapping AVID onto Repercep

The design: how an `AvidEngine` would implement the **same**
`InteractiveWorldModel` seam as V-JEPA 2-AC, but with `step` running a **cached
conditioned denoise** that reuses `denoise_cosmos_video` and emits a pixel frame.
Everything here is **(design)** except the real pieces it builds on, which keep
their `file:line`.

## 2.1 The seam is already the right shape

`AvidEngine` implements `InteractiveWorldModel` (`runtime/interactive.py`):
`reset → step(action) → plan`. Because that Protocol is **shared** with V-JEPA
2-AC, the serving layer needs **zero changes** — `/v2/world/session`
(`serving/app.py`) drives *any* `InteractiveWorldModel`. AVID just plugs in:
`create_app(interactive_engine=AvidEngine(...))`. One seam, two families — the
ADR-0008 point. The entire AVID-specific story lives *inside* `step`.

## 2.2 `WorldState` for a pixel model

V-JEPA's `WorldState.context` held a latent *embedding* window (V-JEPA Part 3).
AVID's holds the recent **diffusion latents** (the conditioning history the next
denoise needs). **Same `WorldState` type** (`runtime/types.py`), different
contents — still in-process, never on the wire.

## 2.3 `reset` (design)

```python
def reset(self, conditioning, params):
    frames = self._resolve_frames(conditioning)        # seed image/video (real URI decode = the work)
    latent = self._pipe.vae.encode(frames)             # reuse the Cosmos VAE (REAL; Cosmos Part 6)
    return WorldState(context=_latent_window(latent), step_index=0, session_id=_new_id())
```

Encode the seed observation into the diffusion latent space — the pixel analogue
of V-JEPA's `get_vision_features`.

## 2.4 `step` — a cached conditioned denoise (the crux, design)

```python
def step(self, state, action):
    with torch.inference_mode():
        next_latent = denoise_cosmos_video(            # REAL function + REAL cache (runtime/denoise.py)
            self._pipe,
            num_frames=self._config.chunk_frames,      # short: one chunk per step, not a 121-f clip
            num_inference_steps=self._config.step_iters,
            cache_mode="adaptive",                     # the adaptive cache pays off here (Cosmos Part 3)
            # --- (design) additions to the loop AVID needs: ---
            init_latent=state.context,                 # (design) start from the current state, not pure noise
            action_conditioning=action,                # (design) the adapter hook — §2.5
        )
        frame = self._pipe.vae.decode(next_latent)     # REAL VAE decode (Cosmos Part 6)
    new = WorldState(context=_update_window(state.context, next_latent),
                     step_index=state.step_index + 1, session_id=state.session_id)
    chunk = FrameChunk(frame_index=new.step_index, total_frames=0,
                       height=int(frame.shape[-2]), width=int(frame.shape[-1]), latency_ms=0.0)
    return new, LatentStep(step_index=new.step_index, frame=chunk)   # <- frame POPULATED (vs V-JEPA's None)
```

Two real reuses and the design hooks:

- **Reuses `denoise_cosmos_video` + the adaptive cache** — AVID is the path where
  the cache (which V-JEPA structurally *can't* use) directly cuts per-step cost.
- **Populates `LatentStep.frame`** — the envelope field that is always `None` for
  V-JEPA (V-JEPA Part 3) is exactly what AVID fills; the wire types already
  support it, so no type changes.
- **The `(design)` kwargs** — today `denoise_cosmos_video` starts from pure noise
  and takes no action. AVID needs it to **continue from `init_latent`** (a short
  image/video-to-video denoise) and to thread **`action_conditioning`** to the
  adapter (§2.5). Those two loop extensions are the real code work on the runtime
  side.

## 2.5 Where the action enters (the hook, design)

The action must reach the DiT's noise prediction. Repercep already has the seam:
conditioning/attention flow through diffusers' dispatcher (Cosmos Part 4) and
`CosmosAttnProcessor` (Cosmos Part 2 §2.5). Two injection options:

- a **side adapter** whose output is combined with the **frozen** DiT's noise
  prediction (the AVID recipe, Part 1 §1.2) — cleanest, leaves the base untouched,
  and slots in beside the dispatch seam; **recommended**.
- an **action cross-attention / adaLN** stream added per block — more invasive,
  closer to a fine-tune.

The frozen-base side-adapter route matches AVID and avoids forking the Cosmos
model — the same "register beside diffusers, don't fork it" philosophy as the FP8
bridge (Cosmos Part 4).

## 2.6 Serving and planning

- **Serving: unchanged.** `/v2/world/session` (V-JEPA Part 5) streams `LatentStep`s;
  now each carries a `frame`, so `decode_pixels` is effectively on. The base64
  pixel-payload path the v2 generate route already uses (Cosmos Part 6) carries it.
- **Planning: deferred (by economics, not by the seam).** `plan()` over AVID would
  make each `_rollout_energy` candidate a *full multi-step denoise* — orders more
  than V-JEPA's latent rollout (Part 3). Honest design: AVID's `plan()` raises
  `NotImplementedError`, **or** uses the hybrid surrogate — *plan in V-JEPA latent
  space* (cheap energy-MPC) and *render the chosen actions with AVID*. The shared
  seam makes that hybrid natural.

## 2.7 The `AvidEngine` sketch (design — not in the repo)

```python
class AvidEngine:                       # implements InteractiveWorldModel (design)
    model_name = "avid-cosmos"
    def __init__(self, backend, config=None, *, pipe=None, adapter=None): ...
    def info(self) -> EngineInfo: ...
    def reset(self, conditioning, params) -> WorldState: ...        # §2.3
    def step(self, state, action) -> tuple[WorldState, LatentStep]: ...  # §2.4 (the crux)
    def plan(self, state, goal, horizon) -> Action:
        raise NotImplementedError("planning over pixels is expensive; "
                                  "use V-JEPA latent planning + AVID rendering (Part 3)")
```

Mirror of `VJepa2ACEngine`: injectable `pipe` (frozen Cosmos/Wan) + `adapter`, so
the **plumbing** (reset/step envelope, window management, serving) is CPU-testable
with a stub adapter — exactly as `vjepa2_ac` is — while the trained adapter is the
real Phase-3 deliverable (Part 3).

**Next:** Part 3 — the cost model, what training the adapter actually takes, and
the build order.
