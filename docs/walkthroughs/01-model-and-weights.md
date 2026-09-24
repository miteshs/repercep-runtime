# Part 1 — The model and its weights

From Part 0: a generation is iterative denoising of a *latent volume* by a DiT,
decoded by a VAE, conditioned on a text embedding. This part opens the box —
what `from_pretrained` actually loads, where the bytes live, and the components
that come back.

> Source note: Repercep *wraps* HuggingFace `diffusers` for Cosmos. Line numbers
> below are from **diffusers 0.38.0** (`pipelines/cosmos/pipeline_cosmos_text2world.py`);
> the project pins `diffusers>=0.31` and runs on 0.37.1 — a few lines may drift,
> so we also cite by symbol. Repercep's own loader is `src/repercep/models/cosmos.py`.

## 1.1 What `CosmosTextToWorldPipeline` *is* — six components

The pipeline is a container that `register_modules(...)` binds together
(`__init__`, lines 168-189). Six pieces:

| Component | Class | Role (ML) | Rough size |
|---|---|---|---|
| `text_encoder` | `T5EncoderModel` (T5-11B) | prompt → a sequence of text embeddings (the *conditioning*) | ~11 B params |
| `tokenizer` | `T5TokenizerFast` | prompt → token ids (max 512) | tiny |
| `transformer` | `CosmosTransformer3DModel` | the **denoiser** — the DiT dissected in Part 2 | ~7 B (the headline "7B") |
| `vae` | `AutoencoderKLCosmos` | pixels ↔ latent volume (8× spatial; temporal per config) | ~hundreds of M |
| `scheduler` | `EDMEulerScheduler` | the **sampling rule** — noise prediction → next cleaner latent | no params |
| `safety_checker` | `CosmosSafetyChecker` | text + video guardrail (NVIDIA license) | a few B |

**ML note.** The "7B" everyone quotes is the **transformer only**. The T5
encoder is *bigger*, but it runs **once** per generation (Part 3); the
transformer runs **~72 times** (36 steps × 2 for CFG). That asymmetry is why
every optimization in this codebase targets the transformer, never T5.

**Systems note.** `model_cpu_offload_seq = "text_encoder->transformer->vae"`
(line 163): on an 80 GB card diffusers keeps only *one* of these resident at a
time and swaps the rest to CPU, because together they don't fit. The MI300X's
192 GB is exactly what lets Repercep skip that swap dance — a recurring structural
advantage (ADR-0001, `docs/WAN_ON_MI300X.md`).

## 1.2 `from_pretrained` and where the bytes live

`src/repercep/models/cosmos.py:176`:

```python
pipe = CosmosTextToWorldPipeline.from_pretrained(self._config.repo_id, torch_dtype=dtype)
# repo_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World";  dtype = bfloat16
```

Mechanically, that:

1. **Resolves + downloads** the repo into the HuggingFace cache:
   ```
   ~/.cache/huggingface/hub/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World/
     snapshots/<commit-hash>/
       model_index.json          # the manifest: (library, class) per component
       text_encoder/  *.safetensors (+ *.index.json)
       transformer/   *.safetensors (+ *.index.json)
       vae/           *.safetensors
       scheduler/     scheduler_config.json
       tokenizer/     ...
   ```
   `model_index.json` is the manifest — it tells diffusers to build a
   `CosmosTransformer3DModel` from `transformer/`, a `T5EncoderModel` from
   `text_encoder/`, and so on.

2. **`*.safetensors` sharding.** Big components are split into shards with a
   `*.safetensors.index.json` mapping each tensor name → shard file. safetensors
   is a flat, zero-copy format — a JSON header (`{name: dtype, shape,
   byte-range}`) followed by one contiguous blob, **memory-mapped** on load, so
   "loading" is mostly `mmap` + pointer math, not parsing.

3. **dtype cast + device placement.** `torch_dtype=bfloat16` casts during load;
   `pipe.to(device)` (`cosmos.py:179`) moves it onto the backend's device via
   `backend.torch_device(...)` — the ADR-0003 seam. ~38 GB of weights total.

The manifest is tiny and readable without the weights:

```bash
# on a box where it's cached:
cat ~/.cache/huggingface/hub/models--nvidia--Cosmos-*/snapshots/*/model_index.json
```

## 1.3 What Repercep adds at load time

`CosmosEngine.load()` (`cosmos.py:149-198`) is not a thin pass-through — it does
three real things around `from_pretrained`:

- **Neutralizes the in-pipeline guardrail** (`_disable_cosmos_guardrail()`,
  `cosmos.py:306`). The pipeline force-constructs a `CosmosSafetyChecker` in its
  `__init__` (diffusers 179-180) that breaks device detection; Repercep swaps it
  for a no-op and, when enabled, runs the guardrail *itself* around generation
  (text check before `cosmos.py:246`, face-blur after `cosmos.py:389`). This is
  the difference between "the pipeline imports" and "it actually lands on the GPU."
- **Activates the FP8 attention bridge** (`maybe_activate_from_env()`,
  `cosmos.py:168`) — wiring `REPERCEP_FP8_ATTENTION` into diffusers' attention
  dispatcher. Part 4 is entirely about this.
- **Optionally `torch.compile`s the DiT** with the F15/F18 frame-size safety gate
  (`cosmos.py:180-194`, `_gate_compile_if_needed`).

## 1.4 The scheduler is not weights — it's the sampling rule

`EDMEulerScheduler` has **no parameters**. It holds the **noise schedule** — the
sequence of noise levels (`sigmas`) the latent passes through and the rule for
stepping between them. Two facts you'll use in Part 3:

- `prepare_latents` (diffusers 322-349) seeds the latent as
  `randn(shape) * sigma_max` — pure Gaussian noise scaled to the *largest* sigma.
- before VAE decode, the final latent is **un-normalized** by `sigma_data` (plus
  optional per-channel mean/std) — diffusers 630-644.

EDM (Karras et al. 2022) parameterizes the network to predict the clean sample;
turning that into the next latent is the scheduler's job (Part 3).

## Run it (no weights)

You can't pull 38 GB here, but the component contract is verifiable structurally:

```bash
python - <<'PY'
import inspect
from diffusers import CosmosTextToWorldPipeline
print(inspect.signature(CosmosTextToWorldPipeline.__init__))
PY
# -> (self, text_encoder, tokenizer, transformer, vae, scheduler, safety_checker=None)
```

**Next:** Part 2 opens the `transformer` — the DiT — layer by layer, and runs a
tiny one on CPU to watch the shapes.
