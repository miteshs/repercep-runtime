# ADR-0009 — KV/latent reuse across rollout steps (the structural latency lever)

- **Status:** Implemented for the *growing*-window regime, on both the
  persistent-session (`step()`) and CEM-batched (`plan()`) call paths; the
  *sliding*-window (eviction) regime is **not achievable as an exact
  operation** for this model — GPU-verified 2026-07-11 against the real
  pretrained weights (not assumed; see §"Verification findings" below). The
  `step()` engine-side seam (§"Decision") and real torch-hub predictor
  adapter are implemented and GPU-verified. The CEM-batched path
  (`_rollout_energy_batched_cached`, §"CEM-batched resolution" below) is
  implemented and CPU-parity-tested against the same adapter machinery the
  `step()` path already GPU-verified — real-weights GPU-verify of *this*
  path specifically is still pending, so it ships opt-in
  (`plan_batched_kv=False` by default). The eviction case is **incorrect by
  construction**, not merely unbuilt, and inherits into the CEM-batched path
  too: the growing-window precondition is checked over the *whole* rollout
  horizon there, not just one step.
- **Date:** 2026-07-11
- **Relates to:** ADR-0008 (interactive world-model seam),
  `docs/LEVERS_2026_07_H100.md` (batching + bf16, GPU-verified same day),
  `docs/repercep-cem-batching-finding` memory (this lever ranked #1, June 26).

## Context

`VJepa2ACEngine.step()` and `_rollout_energy_batched()` feed the **entire**
context window (`context_frames` frames, `context_frames * tokens_per_frame`
tokens — 2048 tokens at the default 8 frames) through the AC predictor's full
transformer stack on **every** call, even though only one new frame's worth of
tokens is actually new. Today's batching (1.6–2.1×, then 7.3× combined with
bf16 — `docs/LEVERS_2026_07_H100.md`) cuts the *number* of forwards and their
*dtype* cost; it does not cut the *quadratic-in-window* cost of each forward.
This ADR designs the lever that does: standard incremental (KV-cache)
attention, adapted to this model's specific RoPE + sliding-window shape.

### The real predictor's architecture (verified against the source, 2026-07-11)

Fetched from `facebookresearch/vjepa2` (`src/models/ac_predictor.py`,
`src/models/utils/modules.py`) — not paraphrased from memory:

- **Token layout, per frame:** `[action_token, state_token, visual_patches...]`
  (`cond_tokens=2`, no extrinsics in our config), frames concatenated
  sequentially: `x = torch.cat([a, s, x], dim=2).flatten(1, 2)`.
- **Attention:** plain `F.scaled_dot_product_attention(q, k, v, attn_mask=...)`
  — **no native KV-cache, no `past_key_values`, no `use_cache` kwarg anywhere**
  in `VisionTransformerPredictorAC.forward(x, actions, states, extrinsics=None)`.
- **Block-causal mask** (`build_action_block_causal_attention_mask`, verbatim):
  ```python
  def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
      N_T = add_tokens + (H * W)
      N = T * N_T
      mask = torch.zeros(N, N).bool()
      mask_block = torch.ones(N_T, N_T).bool()
      for t1 in range(T):
          for t2 in range(max(0, t1 - T + 1), t1 + 1):
              mask[t1*N_T:(t1+1)*N_T, t2*N_T:(t2+1)*N_T] = mask_block
      return mask
  ```
  Frame `t`'s tokens (including its own action/state) attend to frames `<= t`
  only (`t2` ranges up to and including `t1`). **A frame's K/V never depend on
  any later frame.** This is the fact that makes any caching valid at all.
- **Positions are window-relative, recomputed fresh every call:**
  `T = N_ctxt // (grid_h * grid_w)` — derived from the *current* input's own
  token count, not a caller-supplied offset. There is no `start_pos`/`offset`
  parameter anywhere in the signature.
- **RoPE is rotate-half, standard and exactly invertible/composable**
  (verbatim, per axial band — the same function runs independently on the
  d/h/w position components our `_AcPredictorAdapter` doesn't split further
  since context rows are already flattened patch tokens):
  ```python
  def rotate_queries_or_keys(x, pos):
      omega = torch.arange(D // 2, dtype=x.dtype, device=x.device) / (D / 2.0)
      omega = 1.0 / 10000 ** omega
      freq = torch.einsum("..., f -> ... f", pos, omega)
      emb_sin, emb_cos = freq.sin().repeat(...,2), freq.cos().repeat(...,2)
      y1, y2 = x.unflatten(-1, (-1, 2)).unbind(-1)
      y = torch.stack((-y2, y1), dim=-1).flatten(-2)
      return x * emb_cos + y * emb_sin
  ```
  This *would be* the standard complex-rotation form — `RoPE(x, pos) = R(pos) x`
  with `R(pos)` a block-diagonal rotation by angle `pos * ω_i` per frequency
  band `i`, composable as `R(pos - 1) = R(-1) · R(pos)` — **if the code above
  paired frequencies correctly. It does not (see §"GPU verify" below,
  Finding 1): the real model's `.repeat(...,2)` tiles rather than interleaves
  the per-pair frequencies, so `R(a)·R(b) ≠ R(a+b)` in general here.** This
  section's original claim — that a rotated key can be cheaply re-expressed
  at a shifted position via a fixed `R(-1)` — is **wrong for this model** and
  is kept here (struck through in spirit, not in text) to show the reasoning
  that GPU verification overturned; do not reuse the `R(-1)`-shift idea
  without re-deriving it against the actual RoPE implementation in hand.

## The two reuse opportunities (both real, both grounded in the mask above)

1. **Temporal (within/across `step()` calls):** while the window is *growing*
   (session hasn't hit `context_frames` yet), each already-processed frame's
   K/V is exactly reusable — nothing shifted. Net cost per step: **one
   forward over the new frame's tokens** (query against the full cached
   window) instead of a full window forward — O(window) instead of O(window²)
   in the attention term. **GPU-verified real, real weights.** Once the
   window is *full* and would need to FIFO-evict the oldest frame, this stops
   being free — see §"GPU verify" Finding 2: eviction is not recoverable from
   the K/V cache alone for a full-depth causal transformer, regardless of the
   RoPE question. The growing-window win and the eviction problem are
   separate facts; the original draft of this section conflated them.
2. **Across CEM candidates (within one `plan()`'s batched rollout):** because
   frame `t`'s K/V cannot depend on frame `t+1`'s action token (causal mask,
   confirmed above), the **pre-rollout context window's K/V is identical
   across all `S` candidates** — it can be computed **once**, not once per
   candidate (today's `_rollout_energy_batched` computes it redundantly
   inside the batched SDPA call, S times). Candidates only diverge from the
   first *new* frame onward.

## Decision

Extend the `_Predictor` seam with an **optional** incremental-call contract,
additive to the existing full-context callable (nothing about the batching/
bf16 work changes; a predictor that doesn't support it just keeps using the
full-recompute path — same pattern as `supports_batch`):

```python
class _CachedPredictor(Protocol):
    supports_kv_cache: bool  # capability flag, mirrors supports_batch

    def init_cache(self, context: torch.Tensor) -> Any:
        """Build a cache (opaque to the engine — predictor-owned per-layer
        K/V, however it wants to represent them) from a full context window,
        once, at session start / whenever the engine has no cache yet."""

    def step_cached(self, cache: Any, action: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Attend one new frame's tokens (derived from `action`, predictor's
        own concern) against `cache`, return the predicted next frame and the
        cache with that frame appended."""

    def evict(self, cache: Any) -> Any:
        """Drop the oldest frame. NOT a free operation for a full-depth
        causal transformer: retained frames' deeper-layer hidden states are
        already contaminated by having attended to the evicted frame,
        irrecoverably from K/V alone — GPU-verified (§"GPU verify" Finding 2:
        layer 0 matches a fresh recompute exactly, layer 1+ diverges ~11x
        immediately). Do not implement this against real weights without
        first deciding how to handle that (approximate-and-measure,
        attention-sinks, or full-recompute-on-evict)."""
```

`branch(cache) -> Any` (a cheap independent fork, for the CEM-batched case) is
part of the design but not implemented — see the scope note below.

**What is implemented and CPU-tested now** (this session, `step()` path
only): `VJepa2ACEngine` detects `supports_kv_cache` and, when present, drives
`step()` through `init_cache`/`step_cached`/`evict` instead of the
full-window `_predictor(context, action)` call, keeping one cache per
session (mirrors the `_plan_mean` per-session dict `plan_warm_start` already
added). A **fake** cached predictor (toy linear dynamics where the output
depends on the *sum of all live cached frames* — sensitive enough that a
dropped, duplicated, or stale frame changes the result) proves the
bookkeeping (growth vs. eviction) produces results **identical** to the
existing full-recompute path across many steps, including past the
`context_frames` cap where eviction kicks in — the parity test that matters,
same discipline as the June `_rollout_energy_batched` vs. `_rollout_energy`
parity test. **Caveat added post-GPU-verify:** this test validates that the
*engine* calls `init_cache`/`step_cached`/`evict` correctly *given a
predictor whose `evict` is exact* — the fake predictor's toy `evict` (drop
one entry from a list) trivially is exact, by construction. It does **not**
demonstrate that a real predictor's `evict` can be exact — §"GPU verify"
Finding 2 shows it cannot, for this model. The engine seam is sound; the
assumption that any predictor could satisfy it exactly was not.

**Scope note — why `_rollout_energy_batched` (CEM) is design-only for now:**
that path calls the predictor once per rollout timestep with **all S
candidates stacked into one batched tensor** (the batching lever). True
per-candidate KV-caching there means each of the S candidates needs its own
cache once they diverge (from the second rollout step on), which the current
one-big-tensor batched call can't express — it needs either S independent
`step_cached` calls (defeating the batching win) or a paged-attention-style
design where S candidates' caches live in one addressable block and a single
batched kernel call still attends each candidate against only its own cache.
The latter is the right answer long-term but is substantially more engine
machinery than the persistent-session case; deferred as a named follow-up
rather than built partially. **The `step()` win (this ADR's implemented part)
matters more for real serving anyway** — it's what metric #1 of
`docs/CONTROL_LOOP_BENCH.md` (closed-loop step latency under state carryover)
directly measures, and is the online robot-control-loop cost, not just the
offline-planning cost.

## CEM-batched resolution (2026-07-13)

The "paged-attention-style machinery" the scope note above worried about
turns out to be unnecessary for CEM specifically, because CEM's rollouts are
**lockstep-uniform**: all `S` candidates advance exactly `H` horizon steps
together, so per-candidate divergence is a dense `(S, ...)` batch, not a
ragged structure needing a page table. `branch(cache) -> Any` (the design-only
primitive from §"Decision") is not needed either — the resolution below
reuses the existing `_CachedPredictor` methods unchanged, generalized to
accept a batch dimension.

**The mechanism — reuse, not new methods.** `_AcKVCache`'s `layer_kv` and
`pending` tensors already carry a leading batch dimension; it was always 1 in
the `step()` path. `_AcPredictorAdapter._advance_pending` (which
`step_cached` calls) now detects when `action` is `(S, action_dim)` instead
of `(action_dim,)`, and **expands — not copies —** the batch-1 prefix K/V and
pending frame up to `S` via `.expand()` (a view) at the point where a batched
caller first uses them. `torch.cat([k_old, k_new], dim=2)` (concatenating the
expanded prefix with the batch-S new-frame K/V along the *token* dimension,
not the batch dimension) is what turns the cache genuinely batch-`S` from
that point on — `cat` always allocates, so the expanded view's zero-stride
trick only has to survive one call, not the whole rollout. `step_cached` and
`append_frame` needed no new methods, only shape generalization (`action`
`(S, A)` in, `predicted` `(S, P, D)` out; `normed_block` `(S, P, D)` in for
`append_frame`) — the *same* two-pass real-action/zero-action-commit logic
that `step_cached`'s docstring already documents for the single-session case
applies unchanged per-candidate.

**The engine side — `VJepa2ACEngine._rollout_energy_batched_cached`.** Given
`state` and `(S, H, A)` candidate sequences: checks the growing-window
precondition over the **whole horizon** (`frames_now + H <= context_frames`
— stricter than `step()`'s single-step check, since all `S` candidates must
stay eviction-free for all `H` steps, not just one), returns `None` if it
doesn't hold (never a wrong answer — `_candidate_energies` falls back to the
existing `_rollout_energy_batched`, exactly `step()`'s own fallback
discipline for its cache), otherwise builds one shared-prefix cache via
`init_cache(state.context)` and loops `step_cached` → `append_frame` for `H`
steps, computing the terminal energy from the last predicted, normed block —
the same energy formula `_rollout_energy_batched` uses.

**Cost.** Where `_rollout_energy_batched` recomputes the full `O(window)`
prefix attention inside the batched SDPA call at every one of the `H`
timesteps (`S` times over, since the batch dim doesn't change that), the
cached path pays the prefix cost once (`init_cache`, batch 1) and each
subsequent step is a single-frame query per candidate against the shared
cache — the same `O(window²) -> O(window)` asymptotic win the `step()` path
already has, now applied per rollout inside `plan()` too.

**Verification — CPU parity only so far, opt-in until GPU-verified.** Two
tiers, mirroring how the `step()` path's own KV cache was verified before it
was trusted on real weights:
1. *Adapter level* (`test_ac_predictor_adapter_batched_step_cached_matches_per_candidate_loop`):
   `S` candidates run through the batched path in one shared cache must match
   running the same miniature real-RoPE-structure predictor's
   `step_cached`/`append_frame` `S` times independently — proves the
   expand-not-copy batching doesn't mix candidates' K/V. Deliberately broken
   twice while writing it (forgot the expand entirely; expanded the wrong
   tensor — `v_old` in place of `k_old`) to confirm the test actually catches
   both a crash-class and a silent-wrong-value-class regression, not just the
   happy path — same discipline as the `step()` path's original
   branch-safety test.
2. *Engine level* (`test_rollout_energy_batched_cached_matches_uncached_batched_path`,
   `..._falls_back_past_growing_window`, `..._disabled_by_default`): a toy
   batched+cacheable predictor (running-sum dynamics, same "wrong sum = wrong
   answer" sensitivity as the `step()` path's `_KVCacheFakePredictor`) proves
   the engine's dispatch, growing-window precondition, and default-off gating
   are correct, independent of the real RoPE math (already proven at tier 1).

This is CPU-parity evidence the *algorithm and its integration* are correct
against known-good references, exactly what the `step()` path's CPU tests
were before its GPU verify — not yet evidence the real predictor's actual
forward matches under this batching. `plan_batched_kv` therefore defaults to
`False`; flip it only after a real-weights GPU verify (mirroring
`scripts/verify_kv_growing_window.py`'s methodology, extended to a batch of
candidates) confirms energies match the uncached batched path within fp32
tolerance, the same bar `step()`'s cache cleared before its default flipped.

## Verification findings (2026-07-11)

Findings 1–2 came from wrapping the real torch-hub predictor (`facebookresearch/vjepa2`,
`vjepa2-ac-vitg.pt` checkpoint) directly, using its own verbatim source (a
temporary `F.scaled_dot_product_attention` monkeypatch confirmed a
hand-written replica of `ACRoPEAttention`'s pre-SDPA Q/K/V computation is
bit-exact against the live module — `q/k/v` match to `0.0` max abs diff — so
the incremental adapter below reuses provably-correct building blocks, not
guesses). Full end-to-end multi-layer replica of `predictor.forward()` also
matches the real forward exactly (`0.0` diff). Finding 3 came from integrating
that algorithm with Repercep's adapter contract and is covered by CPU parity
tests over the same multi-layer token/attention structure.

**Finding 1 — the model's own RoPE is not a composable rotation.** The
source carries a maintainer comment: *"This expansion has a subtle bug where
frequencies are duplicated across the vector pair... fixing it would break
compatibility with the pretrained model."* Concretely, `rotate_queries_or_keys`
tiles (not interleaves) its per-pair frequencies, so the two components of
each rotated pair get *different* angles — `M(pos)` is linear in `pos` but is
**not** `R(pos)` for any single rotation `R`, so `M(a)·M(b) ≠ M(a+b)` in
general. The `R(-1)`-composition shift this ADR's "Decision" section
originally specified is therefore invalid for the real model (it was derived
from how a *textbook* RoPE would behave, not this one). **Fix:** cache the
*pre-rotation* K (and V, which was never rotated) instead of the rotated K,
and re-derive the rotation fresh at the correct window position on every use.
This is still O(window) per step (an elementwise op, not a matmul) — the
asymptotic win survives. Verified bit-exact (`0.0` diff) against a fresh
subwindow recompute, in isolation, at every layer.

**Finding 2 — eviction is not recoverable at the K/V level, at any layer
depth, for a full-depth causal transformer.** This is the one that actually
breaks the design. Comparing frame 1's *raw* (pre-rotation) K — sliced out of
a full 4-frame forward pass vs. independently recomputed as the first frame
of a fresh 3-frame window — layer by layer: **layer 0 matches exactly (`0.0`
diff)**, but **layer 1 diverges immediately (~11× relative error) and stays
divergent through layer 23** (sampled at layers 0, 1, 2, 5, 10, 15, 20, 23 —
all of 1+ show large, non-decaying error). The reason: layer 0's raw K is a
pure per-token linear projection (no cross-token mixing yet), so it's
identical either way. But frame 1's *hidden state* going into layer 1 is
layer 0's *output* — and in the full 4-frame run, frame 1's tokens were
causally allowed to attend to frame 0 at layer 0, mixing frame-0 information
into frame 1's residual stream. Once frame 0 is evicted, there is no way to
recover "what frame 1's layer-1 input would have been if frame 0 had never
existed" from anything stored in a K/V cache — the contamination happened in
the *residual stream*, not in an attention key. **This is not a bug to fix;
it is the well-known hard problem sliding-window LLM serving (StreamingLLM,
attention-sinks, etc.) exists to work around, encountered here freshly.**

**Finding 3 — the durable prefix must use zero historical actions to preserve
the adapter's public semantics.** `_AcPredictorAdapter.__call__` supplies the
requested action only at the trailing frame and deliberately supplies zeros at
all earlier frames. Persisting the trailing frame's action-conditioned K/V
would therefore make cached and uncached multi-step rollouts different models:
the discrepancy is invisible on step 1 and appears at layer 1+ on later steps.
The implemented adapter keeps one visual frame pending, predicts it with the
live action, and separately advances that frame with a zero action before its
K/V becomes durable. This is two one-frame query passes against the shared
prefix per control step, still O(window) rather than a full O(window²) forward.
A miniature two-layer parity test covers multiple successive steps so this
cannot regress to the tempting but semantically wrong one-pass design.

**Net effect on this ADR's design:** the `step()` seam's *growing*-window
case (§"Decision", `init_cache` → `step_cached`, no `evict`) is real, GPU-
verified end-to-end on pretrained weights (grow-only multi-step test:
`3.8e-4` max abs diff against a from-scratch recompute — within fp32
tolerance, confirmed via the SAME bit-exact-verified building blocks). The
*sliding*-window case (`evict`, triggered once a session exceeds
`context_frames`) is **not** a free, exact operation — implementing it would
mean picking an explicit approximation (e.g., accept the drift and measure
its effect on planning quality; keep an attention-sink prefix per StreamingLLM;
or simply full-recompute on every eviction, forfeiting the speedup for that
regime) — a real design decision requiring its own validation, not a
mechanical follow-up. **`evict()` in the engine-side `_CachedPredictor`
Protocol (§"Decision") should not be wired to a real implementation until
that decision is made explicitly** — until then, `use_kv_cache` only helps
sessions that stay within `context_frames`, and the config should size
`context_frames` to the expected session/plan horizon to get the win without
silently hitting the un-implemented, unsound eviction path.
The real adapter now implements that grow-only design with pre-rotation K and
zero-action durable prefixes; the default `kv_evict_policy="reinit"` drops the
cache at saturation rather than invoking the approximate eviction operation.

**Finding 3's integration re-verified end-to-end on pretrained weights
(2026-07-11/12, MI300X, `scripts/verify_kv_growing_window.py`).** The earlier
`3.8e-4` number above verified the growing-window *algorithm* directly against
a hand-wrapped predictor; this run verifies the actual wired-up
`_AcPredictorAdapter.init_cache`/`step_cached`/`append_frame` contract (the
code CPU-tested against a synthetic predictor above) through
`VJepa2ACEngine.reset()`+real weights, growing a session from 1 to 8 frames.
At `dtype="float32"` (encoder output + action + predicted-frame boundary all
fp32, matching how the original number was measured): max abs diff **6.8e-4**
across 6 steps, same order as the original finding — the CPU-tested
integration is real. **One methodology note, not a regression:** the first
attempt used this engine's `dtype="bfloat16"` default (the *encoder's* output
cast, independent of `predictor_compute_dtype`) and saw max abs diff up to
0.125 — alarming until traced to bf16 boundary rounding on ~100-magnitude
values (bf16 has ~3 significant digits; 0.125 ≈ 100 × 2⁻¹⁰, the right order
for that quantization), not a cached-vs-uncached algorithm divergence. Re-
running at fp32 isolated the algorithm from that rounding and matched the
known-good number. Relative diff (elementwise, vs `expected`) ran as high as
0.29 even in the fp32 run — an artifact of near-zero elements in `expected`
inflating the ratio, not a sign of a large real error; absolute diff (as used
throughout this ADR) is the meaningful metric here.

## Consequences

- The growing-window case genuinely changes the *asymptotic* shape of the
  attention cost from O(window²) to O(window) per step, GPU-verified on real
  weights — a real, usable win for any session/plan whose length fits inside
  `context_frames` (batching and bf16, by contrast, are constant-factor wins
  on top of the existing O(window²) shape).
- The sliding-window (eviction) case does **not** get this win for free; it
  requires a separate, explicit design decision (approximate-and-measure,
  attention-sinks, or accept full-recompute-on-evict) that this ADR does not
  make. Long-running sessions beyond `context_frames` are the primary
  real-world use case this ADR originally targeted (a robot control loop
  running indefinitely) — so the *practical* value of this lever today is
  narrower than the "GPU-verify follow-up" framing this doc previously used
  implied. Sizing `context_frames` to the task's bounded horizon (viable for
  fixed-length manipulation episodes, not for indefinite operation) is the
  only zero-approximation way to use it as designed.
- Session `WorldState.context` (the wire-safe embedding window) is unchanged;
  the cache is engine-held state keyed by `session_id`, mirroring the pattern
  `LingBotVAPipeline` already established for its own named KV cache.
- The CEM-batched dimension (§"CEM-batched resolution") no longer needs the
  paged-attention machinery the original scope note anticipated — CEM's
  lockstep-uniform rollouts (every candidate advances the same `H` steps)
  make the shared-prefix reuse a dense batch expand, not a ragged page table.
  Implemented, CPU-parity-tested, opt-in (`plan_batched_kv=False`) pending
  its own real-weights GPU verify — the growing-window win it delivers is the
  *offline planning* cost (`plan()`), complementary to (not a replacement
  for) the `step()` path's *online control-loop* win.

## Revisit if

*(Original text, kept for the record — its prediction is exactly what
happened):* "The real predictor wrapping proves the shift math doesn't hold
exactly... in which case the fallback is 'cache the growing-window phase
only, full-recompute after first eviction', which still captures most of a
session's early-step savings and is a much smaller, safer change." **This is
now the adopted design** (§"GPU verify"), for a different reason than
anticipated (not a band-split edge case, but the deeper structural fact that
eviction can't be exact for a full-depth causal model at all).

Genuinely revisit the sliding-window (eviction) case if: (a) a real workload
needs sessions longer than a bounded-horizon episode can accommodate inside
`context_frames`, making "just grow the window" impractical (memory or
latency cost of a very large `context_frames`), or (b) there's appetite to
implement and *validate* an explicit approximation (attention-sinks-style
kept prefix, or measured-acceptable drift) rather than treat this as a free
lunch — that is real, separate design + evaluation work, not a bug fix.
