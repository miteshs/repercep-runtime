"""V-JEPA 2 (Meta, Feb 2025) smoke + benchmark on the Repercep backend.

V-JEPA 2 is an *encoder-predictor* world model — Yann LeCun's group's
non-diffusion answer to video generation.  It does NOT decode to pixels.
Inference produces dense embeddings useful for action recognition,
retrieval, planning, and as a vision tower for VLMs.

This script:
  1. Loads V-JEPA 2 (HuggingFace ``facebook/vjepa2-vit{l,h,g}-fpc64-256``)
  2. Runs the encoder on a 64-frame synthetic video clip
  3. Times the forward pass on Repercep's selected backend
  4. Prints embedding shape + timing

It deliberately uses synthetic video so no torchcodec dep is needed.

Usage::

    .venv/bin/python scripts/run_vjepa2.py --model vitl --warmup 2 --iters 5
    .venv/bin/python scripts/run_vjepa2.py --model vitg --resolution 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _setup() -> None:
    """Make src/ importable when running from a worktree."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))


_setup()


import torch  # noqa: E402

from repercep.backend.registry import select_backend  # noqa: E402

_VARIANTS = {
    "vitl": "facebook/vjepa2-vitl-fpc64-256",  # 0.3B
    "vith": "facebook/vjepa2-vith-fpc64-256",  # 0.7B
    "vitg": "facebook/vjepa2-vitg-fpc64-256",  # 1B
    "vitg-384": "facebook/vjepa2-vitg-fpc64-384",  # 1B @ 384 res
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(_VARIANTS.keys()), default="vitl",
                   help="V-JEPA 2 variant (default: vitl, 0.3B)")
    p.add_argument("--backend", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--frames", type=int, default=64,
                   help="frames per clip (V-JEPA 2 was trained on 64)")
    p.add_argument("--resolution", type=int, default=256,
                   help="spatial resolution (256 for fpc64-256; 384 for fpc64-384)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    device = torch.device(backend.torch_device(0))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[repercep] backend={backend.name}  device={device}  dtype={dtype}")
    print(f"[repercep] loading {args.model} ({_VARIANTS[args.model]}) ...")
    t0 = time.perf_counter()
    from transformers import AutoModel

    model = AutoModel.from_pretrained(_VARIANTS[args.model], dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[repercep] loaded in {time.perf_counter() - t0:.1f}s ({n_params / 1e6:.0f} M params)")

    torch.manual_seed(args.seed)
    # Synthetic video: (B, T, C, H, W) in [0, 255] uint8 ish, but transformer
    # processors normalize internally.  We bypass the processor and feed
    # processor-equivalent normalised floats directly to avoid pulling in
    # torchcodec for a synthetic input.
    video = torch.randn(args.batch, args.frames, 3, args.resolution, args.resolution,
                        device=device, dtype=dtype)

    print(f"[repercep] input shape: {tuple(video.shape)}  dtype: {video.dtype}")

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model.get_vision_features(pixel_values_videos=video)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run.
    t0 = time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad():
            out = model.get_vision_features(pixel_values_videos=video)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / args.iters

    peak_gib = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else None

    if isinstance(out, torch.Tensor):
        embed_shape = tuple(out.shape)
    else:
        embed_shape = tuple(out.last_hidden_state.shape)

    print(f"[repercep] encoder forward: {elapsed * 1000:.1f} ms / call  "
          f"(B={args.batch}, T={args.frames}, {args.resolution}x{args.resolution})")
    print(f"[repercep] embedding shape: {embed_shape}")
    if peak_gib is not None:
        print(f"[repercep] peak HBM: {peak_gib:.2f} GiB")

    import json
    result = {
        "model": _VARIANTS[args.model],
        "params_M": round(n_params / 1e6, 1),
        "device": str(device),
        "dtype": str(dtype),
        "batch": args.batch,
        "frames": args.frames,
        "resolution": args.resolution,
        "ms_per_call": round(elapsed * 1000, 2),
        "embedding_shape": list(embed_shape),
        "peak_hbm_gib": round(peak_gib, 2) if peak_gib is not None else None,
    }
    print(f"[repercep] RESULT {json.dumps(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
