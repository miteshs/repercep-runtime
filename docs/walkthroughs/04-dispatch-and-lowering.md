# Part 4 — Dispatch and lowering

Part 2 ended at one line inside the DiT's attention (transformer_cosmos.py:200):

```python
hidden_states = dispatch_attention_fn(q, k, v, attn_mask=mask, is_causal=False)
```

This part follows that call down to a kernel. It's pure **systems** — the seam by
which a vendor-neutral runtime hooks a custom kernel into a library (`diffusers`)
it doesn't control, without forking it. Repercep's code: `src/repercep/attention/
diffusers_backend.py`.

## 4.1 The problem: the model bypasses Repercep's own registry (F19/F40)

Repercep has an attention abstraction (`attention/registry.py`, `select_attention_op`).
But **the Cosmos model never calls it** — `dispatch_attention_fn` consults
diffusers' *own* `_AttentionBackendRegistry`. So Repercep can't intercept attention
by owning the model; it has to **register a backend with diffusers' dispatcher.**
This mismatch is finding **F19**, and its Wan cousin **F40** (the bridge engaged
0× on Wan until a processor fix — `docs/BUILD_LOG.md`). Internalize it: *the hook
point is diffusers' dispatcher, not the model.*

## 4.2 Getting a foothold in a sealed enum (lines 289-339)

diffusers keys backends by an `AttentionBackendName(str, Enum)` and looks them up
in `_member_map_` / `_value2member_map_`. The enum is sealed (you can't add
members the normal way). `_ensure_backend_name_registered` (289) does it anyway,
surgically:

```python
new_member = str.__new__(AttentionBackendName, "repercep_fp8")   # bypass Enum's guard
new_member._name_ = "REPERCEP_FP8"; new_member._value_ = "repercep_fp8"
AttentionBackendName._member_map_["REPERCEP_FP8"] = new_member     # patch the maps
AttentionBackendName._value2member_map_["repercep_fp8"] = new_member  # the lookup reads
```

`register_repercep_fp8_backend` (318) then drops `_repercep_fp8_attention` into
`_AttentionBackendRegistry._backends[member]` and mirrors the supported-arg-name
introspection the official decorator does. It runs **on import** (394) but is
**inert** — the dispatcher stays on `"native"` until something flips it.

## 4.3 Flipping it on (lines 342-388)

`maybe_activate_from_env()` (364) — called from `CosmosEngine.load()` (Part 1,
`cosmos.py:168`) — sets the active backend iff `REPERCEP_FP8_ATTENTION` is set. This
is what makes one env var govern *both* Repercep's own registry and the diffusers
pipeline path. `set_active_backend` mutates a process-wide singleton (the
dispatcher is global), which the code calls out honestly (350-354).

## 4.4 The backend function — the routing decision tree (lines 160-286)

`_repercep_fp8_attention` receives diffusers' `(B, S, H, D)` tensors and decides,
per call, whether the FP8 kernel earns its keep. In order:

1. **`REPERCEP_FP8_ATTENTION=fa`** (193) → route to FlashAttention-2/3 via
   `_native_fallback`, skipping FP8 entirely. This exists because **on Hopper the
   FP8 Triton kernel is 2–3× *slower* than cuDNN-FA3** (F27/F29), but the FA-2/3
   wheel beats both — so on H100 the headline uses `=fa`, not `=1`. (This is the
   FA-3 ×1.39 in the 3.81× decomposition.)
2. **`_parallel_config` set** (199) → fall back (no context-parallel support).
3. **mask / dropout / GQA / fails `_route_should_use_fp8`** (216) → fall back to SDPA.

`_route_should_use_fp8` (52) is the honest "FP8 only sometimes wins" gate:

```python
dtype in (bf16, fp16)           # the kernel casts to FP8 internally
and query.shape == key.shape == value.shape   # SELF-attention only (cross-attn falls out)
and head_dim in {32, 64, 128, 256}            # compiled tile set
and seq_len >= 4096                            # the crossover: 0.53x @4k, 1.92x @8k (Session 9)
```

Consequence worth pausing on: in the Cosmos block (Part 2 §2.4), **only `attn1`
(self-attention, S≈109k) qualifies**; `attn2` (cross-attention to the 512-token
text, Q≠K shape) always falls back to SDPA. FP8 accelerates self-attention only.

4. **Qualifies** → lazy-instantiate the *vendor-correct* op (240): `FP8TritonAttention`
   on `torch.version.hip`, `FP8HopperTritonAttention` on CUDA — siblings that
   **don't cross-compile** because gfx942 is e4m3**fnuz** (max 240) and Hopper is
   e4m3**fn** (max 448) (F25/ADR-0006). Permute `(B,S,H,D)→(B,H,S,D)`, run the
   kernel (Part 5), then a **non-finite guardrail** (271): if FP8 quantization
   produced NaN/Inf, fall back to SDPA *for this call* rather than poison the
   whole denoising step. Permute back.

## 4.5 The default path: `_native_fallback` → SDPA → aotriton (lines 82-157)

When FP8 doesn't apply (the common case, and the entire MI300X default before you
set the env var), `_native_fallback` (82) replicates diffusers' native attention:
permute `(B,S,H,D)→(B,H,S,D)`, call `torch.nn.functional.scaled_dot_product_attention`,
permute back. **On ROCm, SDPA dispatches to aotriton flash kernels** — that's the
working, no-env-var path the headline-without-FP8 numbers run on. (On CUDA it
first tries the cached `HopperFlashAttention` FA-2/3 wrapper, then SDPA→cuDNN.)

## 4.6 The full lowering chain

```
CosmosAttnProcessor2_0  (builds Q,K,V; QK-norm; RoPE; GQA)        Part 2
   └─ dispatch_attention_fn                                       diffusers dispatcher
        └─ active backend:
             ├─ "native"      → SDPA → aotriton flash (ROCm)      ← default kernel
             │                  or cuDNN-FA / FA-2/3 wrapper (CUDA)
             └─ "repercep_fp8"   → _repercep_fp8_attention (routing)
                   ├─ fa mode  → FA-2/3 wheel  (Hopper win)
                   ├─ qualifies→ FP8 Triton kernel               ← Part 5
                   └─ else     → SDPA fallback
```

## Run it (CPU, no weights)

The registration/enum-injection is CPU-testable — `tests/test_attention_diffusers_backend.py`
asserts the `"repercep_fp8"` member resolves, registration is idempotent, the env
bridge flips the active backend, and a short self-attn falls back to native:

```bash
python -m pytest tests/test_attention_diffusers_backend.py -q   # in the [dev] env
```

**Next:** Part 5 — inside the FP8 Triton kernel itself, and how `tl.dot` lowers to
gfx942 MFMA.
