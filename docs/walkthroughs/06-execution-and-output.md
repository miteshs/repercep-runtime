# Part 6 — Execution and output

The loop (Part 3) ended with a **clean latent** — shape `(B, 16, ~31, 88, 160)`,
the denoised version of the noise we started from. This part turns it back into
watchable pixels and streams them out. Code: the tail of `denoise.py` (352-371)
and `CosmosEngine.generate` (`cosmos.py:298-303`).

## 6.1 Un-normalize — back into the VAE's distribution (denoise.py 352-368)

The loop worked in EDM's normalized sigma-space; the VAE expects latents in the
distribution it was trained on. So before decoding, undo the normalization:

```python
if pipe.vae.config.latents_mean is not None:                 # per-channel stats from the VAE
    latents = latents * latents_std / pipe.scheduler.config.sigma_data + latents_mean
else:
    latents = latents / pipe.scheduler.config.sigma_data
```

`latents_mean` / `latents_std` are per-latent-channel constants baked into the VAE
config; `sigma_data` is the EDM scale from Part 1 §1.4. This is the exact inverse
of the seeding (`randn × sigma_max`) + EDM parameterization.

## 6.2 VAE decode — latent → pixels (denoise.py 370)

```python
video = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
```

`AutoencoderKLCosmos` upsamples the latent back to full resolution — **8× spatial
and the temporal factor** — inverting Part 2's encode:

```
latent  (B, 16, ~31, 88, 160)   ──VAE decode──▶   pixels (B, 3, 121, 704, 1280)
```

Systems note: this is the **second-biggest compute** after the DiT, and a **memory
spike** — the whole latent volume expands to the whole pixel volume in one shot
(327 M values at the reference shape). It's a 3D causal-conv decoder. On Wan this
exact step was the **F38 OOM** (a per-frame `torch.cat` in the VAE's forward); the
fix was `--vae-tiling` (decode in spatial tiles). Cosmos has the same shape of
risk; the MI300X's 192 GiB is what absorbs it without tiling (peak ~52.5 GiB
total, the decode being the high-water mark).

## 6.3 Postprocess and the Frame contract (denoise.py 371; cosmos.py 403, 298-303)

`video_processor.postprocess_video(video, output_type="pt")` normalizes to
`(B, T, C, H, W)` in `[0, 1]`. Repercep's `_as_frame_tensor` (`cosmos.py:403`) then
makes it `(T, H, W, 3)` `uint8` on CPU (permute if channels-first, clamp, ×255).
Finally `generate()` yields one `Frame` per index:

```python
total = int(video.shape[0])
for index in range(total):
    yield Frame(index=index, total=total, pixels=video[index])   # cosmos.py:301-303
```

**Honest caveat (cosmos.py:16-19).** This is *post-hoc* iteration over an
already-complete tensor — the whole clip is denoised and VAE-decoded *before* the
first `Frame` is yielded. The `Iterator[Frame]` is a real streaming **contract**,
but for the Cosmos engine it isn't yet denoise-time streaming (the `StubEngine`
*does* stream incrementally). True frame-time streaming is deferred work. So the
`Frame` you saw enter the serving layer in Part 0 is the *end* of a fully-computed
generation, not a frame emitted mid-loop.

## 6.4 Out the wire (serving, app.py:192-213)

Each `Frame` becomes a `FrameChunk` — the metadata envelope (`frame_index`,
`total_frames`, `height`, `width`, `latency_ms`) — serialized as one NDJSON line;
the pixel bytes travel out-of-band (the v2 path base64-encodes them; v1 carries
metadata only). If the guardrail is enabled, a RetinaFace face-blur runs on the
video first and can reject it (`cosmos.py:389`).

The artifact at the very end is a **real 121-frame 1280×704 24 fps H.264 mp4**
(`ffprobe`-verified — `docs/METHODOLOGY.md` §2.5); a single `Frame.pixels` is an
`(H, W, 3)` `uint8` image.

## 6.5 The whole path, end to end

```
GenerationRequest  (prompt, 121f, 36 steps, seed)                       Part 0
  → CosmosEngine.generate → load() (HF cache, 4 components, bf16)        Part 1
  → encode_prompt: T5 runs once → text embeddings                       Part 3 §3.1
  → prepare_latents: randn × sigma_max  →  (B,16,~31,88,160)            Part 3
  → for 36 steps:                                                       Part 3
        scale_model_input → CFG batch-2 DiT forward                      Part 2
            → 28 blocks × {self-attn, cross-attn, FF} (adaLN-zero)       Part 2 §2.4
                → dispatch_attention_fn → SDPA/aotriton or FP8 Triton    Parts 4–5
        → two-call EDM scheduler (CFG on x0) → next latent               Part 3 §3.3
        → adaptive cache may skip the forward entirely                   Part 3 §3.4
  → clean latent → un-normalize → VAE decode → pixels                    Part 6
  → Frame stream → FrameChunk NDJSON                                     Part 6 §6.4
```

Wall-time reality: `generate_seconds` (the headline 142 s MI300X / 99.6 s H100)
**excludes** model load (~10–18 s) and the one-time ~270 s ROCm autotune; it's the
warm steady-state. ~99 % of it is the DiT loop (F4/F10); the adaptive cache (Part
3) is what turns ~470 s of full forwards into ~150 s.

## Run it (the real thing — GPU)

Everything above runs end-to-end on an MI300X/H100 with the weights:

```bash
REPERCEP_FP8_ATTENTION=1 python scripts/run_cosmos.py \
    --frames 121 --steps 36 --native-loop --cache-mode adaptive --cache-adaptive-threshold 0.30
# → an mp4 in benchmark-results/, generate_seconds ≈ 142 (MI300X)
```

---

That's the whole machine, top to bottom. You now know — for Cosmos — where the
weights live, what every layer of the DiT does, how the denoise loop drives it
with CFG and the cache, how an attention call lowers from Python through diffusers'
dispatcher to a gfx942 MFMA instruction, and how the final latent becomes a
streamed frame. The same map applies to Wan (swap the engine + processor) and, for
the *interactive* energy-based path, to V-JEPA 2-AC (ADR-0008).
