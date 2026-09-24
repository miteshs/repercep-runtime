# Part 2 — The DiT, layer by layer

This is the denoiser: `CosmosTransformer3DModel`. One call = one "predict the
noise in this latent" step. We go config → shape journey → `forward()` → one
block → the attention hand-off, on both axes.

> Source: diffusers 0.38.0 `models/transformers/transformer_cosmos.py` (Repercep
> wraps it). Cited by symbol + line; read the real config values on your box with
> `pipe.transformer.config`.

## 2.1 Config — the 7B in numbers

(`__init__`, lines 604-627.)

| Field | 7B value | Meaning |
|---|---|---|
| `num_attention_heads` | 32 | |
| `attention_head_dim` | 128 | ⇒ `hidden_size = 32 × 128 =` **4096** |
| `num_layers` | 28 | transformer blocks |
| `in_channels` / `out_channels` | 16 / 16 | latent channels (match the VAE) |
| `patch_size` | (1, 2, 2) | (t, h, w): no temporal patching, 2×2 spatial |
| `text_embed_dim` | 1024 | T5-11B encoder hidden (the cross-attention dim) |
| `mlp_ratio` | 4.0 | feed-forward expansion |
| `adaln_lora_dim` | 256 | low-rank width of the adaLN conditioning |

The one identity to keep in your head: **`hidden_size = heads × head_dim = 4096`**.

## 2.2 The shape journey (the whole point)

Pixels never enter the DiT — latents do. For the reference config (121 frames,
704×1280, bf16):

```
pixels        (B, 3, 121, 704, 1280)
  │  VAE encode  (8× spatial; temporal per config*)
latent        (B, 16,  ~31,  88, 160)        16 channels; 88 = 704/8, 160 = 1280/8
  │  + padding-mask channel (concat_padding_mask)        → (B, 17, ~31, 88, 160)
  │  CosmosPatchEmbed, patch (1,2,2): each 1×2×2 block of all 17 channels
  │    → a vector of 17·1·2·2 = 68 numbers → Linear(68 → 4096)
tokens        (B, S, 4096)     S = T_p · H_p · W_p = 31 · 44 · 80  ≈  109,000
  │  28 × CosmosTransformerBlock                          (B, S, 4096)
  │  norm_out (adaLN) + proj_out: Linear(4096 → 1·2·2·16 = 64)
              (B, S, 64)
  │  unpatchify  (reshape/permute, lines 802-807)
noise pred    (B, 16, ~31, 88, 160)          ==  latent shape
```

\* The **8× spatial** is certain (88×160 latent → 44×80 post-patch). The
**temporal** factor — and thus the exact token count — comes from the
checkpoint's VAE (`pipe.vae.config.temporal_compression_ratio`; diffusers' bare
fallback is 8). Repercep's writeups quote the production attention shape as
**B = 2 (CFG), H = 32, D = 128, S ≈ 109k** (`docs/ANNOUNCEMENT.md`), which implies
≈ 31 latent frames (temporal ≈ 4). **`S ≈ 109,000` is the number that matters**:
a sequence ~100× longer than a typical LLM prompt. That single fact drives
everything in Parts 4–5.

## 2.3 `forward()` — eight stages

(lines 688-812, in order.)

1. **Padding-mask concat** (702-712): append a validity-mask channel, 16 → 17.
2. **Positional embeddings** (717-719): `CosmosRotaryPosEmbed` (3D RoPE with
   separate temporal/H/W frequency bands, 457-518) yields `(cos, sin)` applied
   *inside* attention; a `CosmosLearnablePositionalEmbed` is added to the tokens.
3. **Patchify** (721-728): `patch_embed` then `flatten(1,3)` → `(B, S, 4096)`.
4. **Timestep embedding** (730-748): `CosmosEmbedding` (69-81) maps the scalar
   timestep → `temb` and `embedded_timestep` (sinusoidal `Timesteps` → SiLU-MLP).
   These condition *every* block (§2.4).
5. **Encoder states** (749-761): the text embeddings (+ optional image context).
6. *(controlnet residual map — skip for the base model).*
7. **Block loop** (772-797): 28× `block(hidden, text, embedded_timestep, temb,
   rope, pos_emb, …)`.
8. **Output** (799-807): `norm_out` (adaLN) → `proj_out` → unpatchify back to a
   latent-shaped noise prediction.

## 2.4 One block — three gated sub-layers

(`CosmosTransformerBlock.forward`, 408-454.) Each block is three residual
sub-layers, and each sub-layer is **adaLN-zero → op → gated residual**:

```python
# 1. self-attention  (tokens attend to tokens)
norm_hidden, gate = self.norm1(hidden, embedded_timestep, temb)   # adaLN-zero
attn = self.attn1(norm_hidden, image_rotary_emb=rope)
hidden = hidden + gate * attn                                     # gated residual
# 2. cross-attention  (tokens attend to the TEXT embedding)
norm_hidden, gate = self.norm2(hidden, embedded_timestep, temb)
attn = self.attn2(norm_hidden, encoder_hidden_states=text, attention_mask=mask)
hidden = hidden + gate * attn
# 3. feed-forward  (4× expand)
norm_hidden, gate = self.norm3(hidden, embedded_timestep, temb)
hidden = hidden + gate * self.ff(norm_hidden)
```

The **adaLN-zero** (`CosmosAdaLayerNormZero`, 114-148) is the conditioning
mechanism — the ML heart of "the timestep tells the layer what to do":

```python
emb = linear_2(linear_1(SiLU(embedded_timestep))) + temb      # low-rank (adaln_lora_dim=256)
shift, scale, gate = emb.chunk(3, dim=-1)
hidden = LayerNorm(hidden) * (1 + scale) + shift              # FiLM-style modulate
# ...the block then multiplies the sub-layer output by `gate` before the residual
```

"Zero" = initialized so `gate ≈ 0`, i.e. each block starts as ~identity and
*learns* how much to contribute — the AdaLN-Zero stability trick from DiT.

Self-attn (`attn1`) gets no `encoder_hidden_states`, so tokens attend to each
other. Cross-attn (`attn2`) passes the **text embedding** as K/V, so every latent
token attends to the prompt. Same `Attention` module, different inputs.

## 2.5 The attention hand-off (bridge to Parts 4–5)

(`CosmosAttnProcessor2_0`, 151-212.) Inside a single attention call:

```python
q = attn.to_q(hidden);  k = attn.to_k(ctx);  v = attn.to_v(ctx)      # projections
q,k,v = [x.unflatten(2,(heads,-1)).transpose(1,2) for x in (q,k,v)]  # split heads
q = attn.norm_q(q); k = attn.norm_k(k)                               # QK-norm (RMSNorm)
q,k = apply_rotary_emb(q, rope), apply_rotary_emb(k, rope)           # RoPE
k = k.repeat_interleave(...); v = v.repeat_interleave(...)           # GQA expand
hidden = dispatch_attention_fn(q, k, v, is_causal=False)             # ← line 200
```

`dispatch_attention_fn` (line 200) is **the seam where "the model" ends and "the
kernel" begins.** On ROCm it lands in SDPA → aotriton; with `REPERCEP_FP8_ATTENTION`
set, Repercep's registered backend routes it to the FP8 Triton kernel. Parts 4 and
5 follow it down.

One detail that changes everything: **`is_causal=False`**. Video attention is
**bidirectional** — every token attends to every token (unlike an LLM's causal
mask). With S ≈ 109k that's a 109k × 109k attention map *per head* — which is why
**flash-style kernels that never materialize that map are mandatory, not an
optimization.**

## 2.6 Where the compute and memory go (systems)

Per block, per forward: self-attention over S ≈ 109k tokens is **O(S²·d)** — the
dominant term; the feed-forward is O(S·d·4d); all ×28 blocks ×~72 forwards
(CFG × steps). In bf16 the activations alone (S × 4096 × 28) are large, which is
why Repercep wraps the loop in `inference_mode()` (no autograd graph — the F18 OOM
fix, ~189 GiB → 52.5 GiB) and why **skipping whole steps** (the adaptive cache,
Part 3) is the single biggest lever. Param count ≈ 7 B (28 blocks × the four
self-attn projections + the two cross-attn + the 4× FF, each ~4096²).

## Run it (CPU, no weights)

A real DiT needs ~14 GB of bf16 weights, but the **structure** runs on CPU at toy
size. This exact snippet, and its real output:

```python
import torch
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
m = CosmosTransformer3DModel(
    in_channels=16, out_channels=16,
    num_attention_heads=2, attention_head_dim=16,   # hidden=32  (7B: 32*128=4096)
    num_layers=2, text_embed_dim=64, mlp_ratio=2.0,
    patch_size=(1,2,2), concat_padding_mask=False,
).eval()
latent = torch.randn(1,16,4,8,8); ts = torch.tensor([500.0]); text = torch.randn(1,8,64)
out = m(hidden_states=latent, timestep=ts, encoder_hidden_states=text, fps=24, return_dict=False)[0]
print(out.shape)
```

Verified output:

```
top-level modules: ['patch_embed','rope','learnable_pos_embed','time_embed','transformer_blocks','norm_out','proj_out']
one block:         ['norm1','attn1','norm2','attn2','norm3','ff']     # the three gated sub-layers
latent=(1,16,4,8,8) -> tokens S = 4*4*4 = 64, dim=32
DiT output (noise pred) = (1,16,4,8,8)     # == input latent shape
```

The output has the **same shape as the input latent** — the invariant of a DiT
step: noisy latent in, same-shaped prediction out. Swap in the 7B config (32×128,
28 layers, `text_embed_dim=1024`) and it's the real thing.

**Next:** Part 3 — the denoise loop (`runtime/denoise.py`), where this `forward`
gets called 36× with CFG and the adaptive cache.
