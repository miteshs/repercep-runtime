# Part 5 — Serving the closed loop, and what's next

The Cosmos path streams a finished clip over NDJSON (Cosmos Part 6). That can't
serve an *interactive* world model — there's no channel to send an action
mid-stream. So Repercep adds a **bidirectional WebSocket**: `serving/app.py`
`/v2/world/session`.

## 5.1 The WebSocket session (serving/app.py)

```python
@app.websocket("/v2/world/session")
async def world_session(ws):
    await ws.accept()
    engine = active_interactive
    if engine is None:
        await ws.send_json({"error": ...}); await ws.close(1008); return
    reset = ResetRequest.model_validate_json(await ws.receive_text())       # 1st message: open
    state = await run_in_threadpool(engine.reset, reset.conditioning, reset.params)
    await ws.send_text(LatentStep(step_index=0).model_dump_json())          # ack the reset
    while True:
        action = Action.model_validate_json(await ws.receive_text())        # client is in the loop
        state, step = await run_in_threadpool(engine.step, state, action)
        await ws.send_text(step.model_dump_json())                          # one LatentStep per action
```

Protocol: the client sends a `ResetRequest`, then one `Action` per step; the
server returns one `LatentStep` per step (`step_index=0` acks the reset). The
**`state` persists in the handler scope** for the life of the connection — the
defining difference from the stateless one-shot path. `WebSocketDisconnect` ends
it and the `WorldState` is GC'd.

## 5.2 Two systems details

- **`run_in_threadpool`.** `engine.reset/step` are sync and multi-millisecond;
  running them in a threadpool keeps the asyncio event loop free — the same
  rationale as the Cosmos v2 driver thread. State is threaded through the handler
  (`state, step = await …`), never shared, so there's no cross-request mutation.
- **Wired via `create_app(interactive_engine=...)`** — `None` (the default)
  disables the route, so adding the seam didn't touch the existing v1/v2 endpoints
  (purely additive). `tests/test_serving_interactive.py` drives the whole loop
  end-to-end through FastAPI's `TestClient` against a stub engine (CPU-verified).

## 5.3 What the real thing still needs

The seam, rollout, planner, and serving are real and CPU-tested. To make it *do*
something on hardware:

1. **The weight port** (Part 2): load the real V-JEPA 2 encoder (HF) and wire the
   AC predictor head (`facebookresearch/vjepa2`) into `_load_ac_predictor`. Then
   `run_vjepa2_ac.py` *without* `--stub` runs on a GPU.
2. **Goal embeddings.** `plan` needs a `goal` in embedding space — typically
   `encoder.get_vision_features(goal_image)`. A goal-image → goal-embedding path
   is the natural next API addition.
3. **Batched CEM** (Part 4 §4.4): roll the `plan_samples` candidates as one batch
   — the obvious throughput win.

## 5.4 The AVID sibling — the same seam, for pixels

ADR-0008's Phase 3: an **AVID-style** engine (action-conditioned *video
diffusion*) can implement the *same* `InteractiveWorldModel` Protocol but **decode
pixels per step** — populating `LatentStep.frame`, reusing the Cosmos denoise loop
+ adaptive cache. That's why the seam matters beyond V-JEPA: it serves **both** a
latent world model (V-JEPA 2-AC, this series) and a pixel one (AVID), under one
interface. AVID needs a *trained* action adapter, so it's deferred — a research
task, not a code stub.

## 5.5 The whole V-JEPA 2-AC path, end to end

```
ResetRequest → reset: encode seed (V-JEPA 2 encoder) → WorldState (latent window)   Parts 2-3
  → loop:  Action → step: AC predictor(context, action) → next latent → window      Part 3
           (each step streamed as a LatentStep over the WebSocket)                   Part 5
  → plan(goal, H): CEM over action sequences, minimize ||rollout_terminal − goal||   Part 4
  → best Action                                                                       Part 4
```

**Real + CPU-tested:** the seam, the rollout, the energy-MPC planner, the serving
loop. **The port:** the V-JEPA 2 encoder + AC predictor weights (GPU). The
contrast with Cosmos holds throughout — **latent not pixels, autoregressive not
denoising, planning not caching** — which is exactly why ADR-0008 built the seam
this session.

That's the V-JEPA 2-AC machine, top to bottom — the energy-based world model,
served.
