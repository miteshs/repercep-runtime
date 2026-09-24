# Part 2 — The model and its weights

Part 1 was the idea (predict embeddings; energy = embedding distance; non-
contrastive training). This part is the *actual model* — the V-JEPA 2 encoder and
the two kinds of predictor — and exactly what's real vs. a port in Repercep's
`models/vjepa2_ac.py`.

> Source: `transformers 5.9.0` `models/vjepa2/modeling_vjepa2.py` +
> `configuration_vjepa2.py` (the version the project runs). Cited by symbol +
> line. Repercep's engine is `src/repercep/models/vjepa2_ac.py`.

## 2.1 `VJEPA2Model` = encoder + predictor (verified)

The HuggingFace model is two transformers (modeling_vjepa2.py:890):

```python
class VJEPA2Model(VJEPA2PreTrainedModel):
    def __init__(self, config):
        self.encoder   = VJEPA2Encoder(config)     # the representation network
        self.predictor = VJEPA2Predictor(config)   # the JEPA masked-prediction head
```

A tiny one instantiated on CPU confirms the structure:

```
top-level:          ['encoder', 'predictor']
encoder:            ['embeddings', 'layer', 'layernorm']
one encoder layer:  ['norm1', 'attention', 'drop_path', 'norm2', 'mlp']
predictor:          ['embeddings', 'layer', 'layernorm', 'proj']
num_patches N = (4/2)*(16/8)^2 = 8
get_vision_features: input (1, 4, 3, 16, 16) -> embeddings (1, 8, 32)
```

## 2.2 The encoder — a ViT over 3D patches

`VJEPA2Encoder` (421): `embeddings → 24 × VJEPA2Layer → LayerNorm`. Walk it:

- **3D patchify** (`VJEPA2PatchEmbeddings3D`, 84): a single **`nn.Conv3d`** with
  kernel = stride = `(tubelet_size, patch_size, patch_size)` = `(2, 16, 16)`. It
  cuts the video `(B, C, T, H, W)` into non-overlapping **tubelets** (2 frames ×
  16×16 px) and projects each to `hidden_size`. So unlike Cosmos (a `Linear` over
  flattened patches), V-JEPA uses a strided 3D conv — same idea, different
  spelling. Token count:
  ```
  N = (frames/tubelet) · (crop/patch)²  =  (64/2)·(256/16)²  =  32·256  =  8192 patches   (ViT-g)
  ```
- **Block** (`VJEPA2Layer`, 373): standard **pre-norm** ViT block —
  `x += attn(norm1(x)); x += mlp(norm2(x))`. No adaLN, no timestep conditioning
  (it's not a diffusion model — there's no timestep). Contrast Cosmos's adaLN-zero
  gated blocks (Cosmos Part 2 §2.4): V-JEPA blocks are plain residual ViT blocks.
- **3D RoPE attention** (`VJEPA2RopeAttention`, 207): self-attention with
  rotary embeddings split across the **three axes** — each token's `(frame,
  height, width)` position is computed (245-277) and RoPE is applied to disjoint
  slices of the head dim (`d_dim`/`h_dim`/`w_dim`, 238-294). `is_causal = False`
  (243) — the encoder sees the **whole clip bidirectionally**.

Config defaults are ViT-L (`hidden_size=1024`, `num_hidden_layers=24`,
`num_attention_heads=16`, `patch_size=16`, `crop_size=256`, `frames_per_clip=64`,
`tubelet_size=2`; configuration_vjepa2.py:64-86). The checkpoint Repercep uses,
**`facebook/vjepa2-vitg-fpc64-256`**, is the **ViT-g** override (~1408 hidden,
~40 layers — ~1 B params). Read the real values with `model.config`.

**`get_vision_features`** (968) is the one entry point Repercep uses:
```python
def get_vision_features(self, pixel_values_videos):
    return self.forward(pixel_values_videos, skip_predictor=True).last_hidden_state  # (B, N, D)
```
It runs the encoder and **skips the predictor** — exactly what `scripts/run_vjepa2.py`
calls, and what Repercep's `reset()` calls to embed the seed observation (Part 3).

## 2.3 The masked-prediction predictor — Part 1's energy, made literal

`VJEPA2Predictor` (551) is the **self-supervised training head** — the thing that
makes JEPA's energy concrete. It takes the encoder's context tokens, inserts
learned **mask tokens** for the positions to predict (`VJEPA2PredictorEmbeddings`,
481), runs its own (smaller) transformer, and projects back to `hidden_size`.

The payoff is in `VJEPA2Model.forward` (942-954):
```python
predictor_output.last_hidden_state              # ŝ_y : PREDICTED target embeddings
target_hidden_state = apply_masks(sequence_output, target_mask)   # s_y : ACTUAL target embeddings
```
Part 1's energy `E = ‖ŝ_y − s_y‖` is *literally* the distance between those two
tensors. Training minimizes it; the actual-target side comes from an EMA target
encoder (the collapse fix, Part 1 §1.3). **This predictor is NOT
action-conditioned** — it predicts *masked* patches of the *same* clip. It's how
the encoder learns good representations, not how you roll a world forward.

## 2.4 The action-conditioned predictor — the part that makes it a *world model*

The piece that turns this into a controllable world model is a **different**
predictor: **V-JEPA 2-AC**. It is **not in HuggingFace `transformers`** (which
ships only the encoder + the SSL predictor above). It lives in
`facebookresearch/vjepa2`: a ~300 M-param, **block-causal** transformer that
**autoregressively predicts the next state embedding conditioned on an action**
and the previous states. Block-causal (causal over time) is what lets you roll it
forward one step at a time — the basis of `step()` and planning (Parts 3–4).

In Repercep this is the **port**: `models/vjepa2_ac.py` `_load_ac_predictor(config)`
raises `NotImplementedError` with the intended wiring in its docstring; the
contract it must satisfy is a callable `(context, action) → next_state_embedding`
(the `_Predictor` Protocol in that file).

## 2.5 How Repercep wires it (models/vjepa2_ac.py)

```python
DEFAULT_ENCODER_REPO   = "facebook/vjepa2-vitg-fpc64-256"   # the HF encoder above (real)
DEFAULT_PREDICTOR_REPO = "facebookresearch/vjepa2"          # the AC head (port)

def _ensure_encoder(self):   # intended: AutoModel.from_pretrained(...).get_vision_features path
    ...                      # proven in scripts/run_vjepa2.py
def _ensure_predictor(self): # the port
    self._predictor = _load_ac_predictor(self._config)      # -> NotImplementedError today
```

So Repercep's `VJepa2ACEngine` = **HF encoder (real, loadable) + the AC predictor
head (the port)**. Everything *above* the model — the rollout, the energy, the
planner (Parts 3–4) — is real and CPU-tested against an injected stub predictor.

## 2.6 Where the weights live

Same mechanics as Cosmos (that series' Part 1): `AutoModel.from_pretrained(
"facebook/vjepa2-vitg-fpc64-256")` pulls the encoder into
`~/.cache/huggingface/hub/...` as safetensors, memory-mapped, placed on the
backend device in the configured dtype. ViT-g is ~1 B params — far smaller than
Cosmos's 7 B DiT, and it runs **once** at `reset()`, not in a loop, so it's not
the cost center here; the per-step AC predictor is.

## Run it (CPU, no weights)

The exact tiny-encoder snippet from §2.1 — instantiate a small V-JEPA 2, run the
encoder, watch the shapes (`transformers` installed, no weights downloaded):

```python
import torch
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Model
from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config
cfg = VJEPA2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                   crop_size=16, patch_size=8, tubelet_size=2, frames_per_clip=4,
                   pred_hidden_size=16, pred_num_hidden_layers=1, pred_num_attention_heads=2)
m = VJEPA2Model(cfg).eval()
video = torch.randn(1, 4, 3, 16, 16)                          # (B, T, C, H, W)
print(m.get_vision_features(pixel_values_videos=video).shape) # -> (1, 8, 32) = (B, N, D)
```

`(B, N, D)` is the latent the world state is built from. Swap in the `vitg-fpc64-
256` config and it's the real encoder.

**Next:** Part 3 — Repercep's `InteractiveWorldModel` seam and `VJepa2ACEngine.reset/
step`, where that `(B, N, D)` becomes a rolling `WorldState`.
