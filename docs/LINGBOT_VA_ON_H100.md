# LingBot-VA 2.0 (base) on H100 — first external measured numbers (2026-07-11)

First measured latency profile of Ant/Robbyant's **LingBot-VA 2.0** — the
video-action world model of `docs/LINGBOT_VA_PORT_PLAN.md` — run end-to-end
(weights → chunked video+action inference → VAE pixel decode) on a RunPod
H100 SXM. Reference stack (their `wan_va` code, `demo_i2av` config), timed by
`va_timed.py` (CUDA-synced per-chunk wall time; scratchpad driver, to land with
the Phase-1 seam port). This is the baseline the Repercep serving work optimizes
against; the MI300X column is the companion run.

## Setup

| | |
|---|---|
| GPU | NVIDIA H100 SXM 80GB (RunPod, $2.99/hr) |
| Image / stack | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` · torch 2.9.1+cu128 · transformers 4.55.2 · diffusers 0.36.0 · numpy 1.26.4 |
| Attention | `attn_mode="torch"` (SDPA) — flash-attn **stubbed**, never called |
| Model | `robbyant/lingbot-va-base` (24.4 GB bundle: T5 11.4G + MoT transformer 10.2G + Wan2.2 VAE 2.8G), bf16 |
| Config | `demo_i2av`: 256x256, 2 cameras, chunk = 4 latent frames = 32 actions (30-dim x 8/frame), attn_window 30 chunks, 5 video + 10 action flow-match steps, CFG 5.0/1.0, 10 chunks |
| Prompt | "Pick the green cube and place it inside the blue box" (repo demo images) |

## Results

| metric | value |
|---|---|
| model load | 8.6 s |
| reset (cache build + prompt + obs encode) | 0.41 s |
| chunk latency, warm mean (chunks 1-9) | **1384.7 ms** |
| chunk latency, cold (chunk 0) | 1544.9 ms |
| synchronous action throughput | **23.1 actions/s** |
| peak HBM, one session | **38.8 GiB** |
| VAE decode (fresh process, 40 latent → 157 px frames) | peak 6.0 GiB |
| actions | all finite; (6 used ch x 40 f x 8); |a| p95 = 134.7 (denorm) |
| rollout sanity | decoded 157-frame video shows the task being executed in imagination (gripper reaches cube → box) |

Verbatim provenance (the `RESULT` line, unedited):

```json
{"model": "robbyant/lingbot-va-base", "config": "demo_i2av", "gpu": "NVIDIA H100 80GB HBM3", "torch": "2.9.1+cu128", "attn_mode": "torch", "dtype": "torch.bfloat16", "load_s": 8.6, "reset_s": 0.41, "chunk_ms": [1544.9, 1250.0, 1510.2, 1379.2, 1407.3, 1370.3, 1347.7, 1380.3, 1396.2, 1421.4], "chunk_ms_warm_mean": 1384.7, "video_steps": 5, "action_steps": 10, "cfg": [5, 1], "actions_per_chunk": 32, "warm_actions_per_s": 23.1, "action_shape": [6, 40, 8], "actions_finite": true, "action_abs_p95": 134.725, "peak_gib": 38.83}
```

## Reading the numbers (vs the paper's Table 3)

The LingBot-VA 2.0 paper's acceleration ladder (their GPU unspecified):
bf16 PyTorch async **927 ms/chunk (35 Hz)** → +consistency distillation 466 →
+FP8 TensorRT 369 → +paged-KV/FlashInfer 272 → +runtime-overhead **142 ms
(225 Hz)**, control frequency = (1000/t_chunk) x 32.

- Our 1385 ms vs their 927 ms baseline is consistent: the released `demo_i2av`
  path runs **CFG 5.0 (doubles every transformer forward)** and SDPA instead of
  flash-attn, and our number is synchronous (no Foresight-Reasoning overlap).
- **Everything below 466 ms in their ladder is CUDA-locked** (FP8 TensorRT
  engines, FlashInfer). The open release ships only the eager bf16 path — so
  the portable optimization layers (paged/ragged KV cache, host-overhead
  amortization, CFG-free or distilled stepping) are exactly the vendor-neutral
  serving gap Repercep's control-regime thesis names. That gap is ~10x.
- **38.8 GiB for a single 256x256 session** (pre-allocated 30-chunk KV window
  at CFG batch 2) means one session per H100 as released. Session-state
  engineering — cache sizing, paging, CFG elimination — is the difference
  between 1 and N resident world-model sessions per GPU; on MI300X's 192 GB
  the as-released number is already ~4.

## Gotchas (all hit, all resolved)

1. Ubuntu-24 base is PEP-668: `pip --break-system-packages` (June lesson, re-hit).
2. `wan_va/modules/model.py` hard-imports flash_attn; `attn_mode="torch"` never
   calls it → a one-line stub module beats a 30-min wheel build.
3. `VA_Server.generate()`'s final full-video `vae.decode` OOMs an 80 GB H100
   (the 38.8 GiB of session state is still resident); decode the saved latents
   in a fresh VAE-only process (`decode_latents.py`, peak 6 GiB).
4. `wan22_pretrained_model_name_or_path` is a placeholder — sed it in
   `wan_va/configs/va_demo_cfg.py` (config is EasyDict; no CLI override).
5. FSDP init needs `env://` even single-GPU: run under
   `torchrun --nproc_per_node=1`.
