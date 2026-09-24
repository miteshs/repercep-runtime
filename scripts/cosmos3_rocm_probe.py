#!/usr/bin/env python3
"""Staged ROCm gate for the Cosmos 3 Nano port (docs/COSMOS3_PORT_PLAN.md §6).

The port's headline claim is "first AMD row on NVIDIA's flagship open physical-AI
policy model." That claim rests entirely on one unverified assumption: that
diffusers' ``Cosmos3OmniPipeline`` has no TransformerEngine / FlashAttention-3 /
apex / CUDA-only custom-op dependency, the way the older Cosmos-Predict1
pipeline did not (see ``repercep/models/cosmos.py``'s module docstring).

So this script is built to **fail cheaply and early**. The stages are ordered by
cost, and the 33 GB weight download does not happen until every structural
question has already been answered:

  0. torch sees the MI300X; bf16 matmul works                    ~seconds
  1. diffusers at the pinned SHA imports; the Cosmos 3 symbols exist ~1-2 min
  2. the pipeline's module tree carries no CUDA-only dependency   ~seconds
  3. the transformer constructs from config on the meta device    ~seconds
     (config files only -- a few KB, not the 33 GB of weights)
  4. real weights + one policy call, timed                        ~33 GB + minutes

Stages 0-3 answer "does this port have a reason to exist?" for the price of a
few minutes of pod time. Only stage 4 costs real money.

Usage on the box:

    python3 cosmos3_rocm_probe.py            # stages 0-3, the cheap gate
    python3 cosmos3_rocm_probe.py --full     # all stages, downloads weights

Each stage prints ``PASS``/``FAIL`` and the script exits non-zero on the first
failure, so it is safe to run unattended and check the exit code.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import subprocess
import sys
import time

# Pinned deliberately: Cosmos3OmniPipeline exists only on diffusers main, and a
# floating main under a benchmark row is how reproducibility dies quietly
# (port plan §2). Bump this consciously and record it in the bench doc.
DIFFUSERS_SHA = "d6726f38a0c5ca6c06a8f227fb7bade3486ed98d"  # main @ 2026-08-08

REPO = "nvidia/Cosmos3-Nano-Policy-DROID"

# Import names that would each kill the ROCm story if the pipeline pulled them.
# flash_attn is listed but treated as a warning rather than a failure: if it is
# imported behind a try/except with an SDPA fallback (the DreamZero situation)
# the port survives, so a hit here needs a human to look rather than an
# automatic verdict.
CUDA_ONLY_MODULES = ("transformer_engine", "apex", "flashinfer", "cudnn_frontend")
SOFT_MODULES = ("flash_attn", "flash_attn_3", "sageattention", "xformers")

# The DROID policy geometry, from run_policy_with_diffusers.ipynb's ACTION_SETS
# + FIXED_SAMPLING. Policy uses flow_shift=5.0 -- forward/inverse dynamics use
# 10.0 and the cookbook README documents only that one (port plan §1.2).
POLICY = {
    "mode": "policy",
    "domain_name": "droid_lerobot",
    "chunk_size": 16,
    "resolution_tier": 480,
    "view_point": "concat_view",
    "fps": 15,
    "prompt": "Pick up the object and place it in the target container.",
    "num_inference_steps": 30,
    "guidance_scale": 1.0,
    "flow_shift": 5.0,
    "seed": 0,
}


def _say(stage: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {stage}" + (f" -- {detail}" if detail else ""), flush=True)


def _die(stage: str, detail: str) -> None:
    _say(stage, False, detail)
    sys.exit(1)


def stage0_torch() -> None:
    """torch sees the GPU and bf16 works. BF16 is not optional here -- the model
    card is explicit that FP4/FP8/FP16 are untested and unsupported."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - box-only path
        _die("stage 0: torch import", str(exc))

    if not torch.cuda.is_available():
        _die("stage 0: torch.cuda.is_available()", "no GPU visible (ROCm torch exposes AMD here)")

    name = torch.cuda.get_device_name(0)
    version = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None)
    is_rocm = getattr(torch.version, "hip", None) is not None
    try:
        a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
        (a @ a).float().sum().item()
    except Exception as exc:  # pragma: no cover - box-only path
        _die("stage 0: bf16 matmul", str(exc))

    stack = "ROCm" if is_rocm else "CUDA"
    _say("stage 0: torch + device", True, f"{name}, torch {torch.__version__}, {stack} {version}")
    if not is_rocm:
        print("       NOTE: CUDA build -- the ROCm gate is not being tested here.", flush=True)


def stage1_diffusers(install: bool) -> None:
    """diffusers at the pinned SHA imports and carries the Cosmos 3 symbols."""
    if install:
        url = f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_SHA}"
        print(f"       installing {url}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _die("stage 1: diffusers install", result.stderr.strip()[-500:])

    try:
        import diffusers
    except ImportError as exc:
        _die("stage 1: diffusers import", str(exc))

    missing = [
        name
        for name in ("Cosmos3OmniPipeline", "CosmosActionCondition")
        if not hasattr(diffusers, name)
    ]
    if missing:
        _die(
            "stage 1: Cosmos 3 symbols",
            f"missing {missing} -- wrong diffusers version? (need main @ {DIFFUSERS_SHA[:12]})",
        )
    _say("stage 1: diffusers + Cosmos 3 symbols", True, f"diffusers {diffusers.__version__}")


def stage2_module_tree() -> None:
    """**The gate the whole port rests on.** Import the pipeline module and check
    what came with it -- no weights, no network beyond the import itself."""
    import diffusers

    before = set(sys.modules)
    pipeline_cls = diffusers.Cosmos3OmniPipeline
    module = importlib.import_module(pipeline_cls.__module__)
    # Pull the transformer/VAE modules in too: the pipeline module alone may not
    # import them until from_pretrained runs.
    for name in ("transformer_cosmos", "autoencoder_kl_cosmos"):
        for candidate in (
            f"diffusers.models.transformers.{name}",
            f"diffusers.models.autoencoders.{name}",
        ):
            with contextlib.suppress(ImportError):
                importlib.import_module(candidate)
    after = set(sys.modules)

    hard = sorted({m.split(".")[0] for m in after} & set(CUDA_ONLY_MODULES))
    soft = sorted({m.split(".")[0] for m in after} & set(SOFT_MODULES))

    if hard:
        _die(
            "stage 2: CUDA-only dependency check",
            f"pipeline imported {hard} -- the ROCm story for this port is dead; "
            "see port plan §6 before spending anything further",
        )

    detail = f"{len(after - before)} modules pulled, no CUDA-only imports"
    _say("stage 2: CUDA-only dependency check", True, detail)
    print(f"       pipeline module: {module.__name__}", flush=True)
    if soft:
        print(
            f"       WARNING: optional fast-attention modules present: {soft}. "
            "Check they are try/except-guarded with an SDPA fallback (DreamZero was) "
            "before trusting a latency number.",
            flush=True,
        )


def stage3_construct_from_config() -> None:
    """Build the transformer from its published config on the meta device.

    Downloads config JSON only (a few KB). Catches "the architecture itself
    needs a CUDA-only op to even instantiate" without paying 33 GB to find out.
    """
    import torch
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(REPO, "transformer/config.json")
    except Exception as exc:
        _die("stage 3: config fetch", f"{exc} (repo is public -- network problem?)")

    with open(path) as handle:
        config = json.load(handle)

    cls_name = config.get("_class_name", "?")
    print(f"       transformer class: {cls_name}", flush=True)
    print(
        "       config keys: "
        + ", ".join(f"{k}={config[k]}" for k in sorted(config) if not k.startswith("_"))[:400],
        flush=True,
    )

    import diffusers

    model_cls = getattr(diffusers, cls_name, None)
    if model_cls is None:
        _die("stage 3: transformer class lookup", f"diffusers has no {cls_name}")

    try:
        with torch.device("meta"):
            model = model_cls.from_config(config)
        params = sum(p.numel() for p in model.parameters())
    except Exception as exc:
        _die("stage 3: meta-device construction", str(exc))

    _say("stage 3: construct from config", True, f"{cls_name}, {params / 1e9:.2f}B params (meta)")


def stage4_real_call() -> None:
    """Full weights + one policy call. This is the expensive stage."""
    import torch
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from PIL import Image

    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        REPO,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=False,  # guardrail is gated; not needed for a latency gate
        # Explicit because the default is environment-dependent: on the H100 box
        # it resolved to False and the load died with "`low_cpu_mem_usage` cannot
        # be False when `keep_in_fp32_modules` is True", while the MI300X box
        # defaulted to True and loaded fine. Pinning it keeps the two rows
        # comparable and the probe portable.
        low_cpu_mem_usage=True,
    )
    pipe.to("cuda")
    load_s = time.time() - t0
    resident = torch.cuda.memory_allocated() / 1024**3
    _say("stage 4a: weights loaded", True, f"{load_s:.1f}s, {resident:.1f} GiB resident")

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=POLICY["flow_shift"], use_karras_sigmas=False
    )

    # A synthetic 640x540 conditioning canvas -- the real three-camera composite
    # is what a benchmark row needs, but for a "does it run at all" gate the
    # canvas geometry is what matters, not the pixels.
    canvas = Image.new("RGB", (640, 540), (32, 32, 32))

    t0 = time.time()
    result = pipe(
        prompt=POLICY["prompt"],
        action=CosmosActionCondition(
            mode=POLICY["mode"],
            chunk_size=POLICY["chunk_size"],
            domain_name=POLICY["domain_name"],
            resolution_tier=POLICY["resolution_tier"],
            image=canvas,
            view_point=POLICY["view_point"],
        ),
        fps=POLICY["fps"],
        num_inference_steps=POLICY["num_inference_steps"],
        guidance_scale=POLICY["guidance_scale"],
        use_system_prompt=False,
        generator=torch.Generator(device="cuda").manual_seed(POLICY["seed"]),
    )
    chunk_s = time.time() - t0

    frames = result.video
    actions = result.action
    peak = torch.cuda.max_memory_allocated() / 1024**3

    if actions is None:
        _die("stage 4b: policy call", "returned no actions -- policy mode must produce a chunk")

    shape = tuple(actions[0].shape) if hasattr(actions[0], "shape") else (len(actions[0]),)
    _say(
        "stage 4b: policy call",
        True,
        f"{chunk_s * 1000:.1f} ms/chunk, {len(frames)} frames, "
        f"actions {shape}, peak {peak:.1f} GiB",
    )
    print(
        json.dumps(
            {
                "repo": REPO,
                "diffusers_sha": DIFFUSERS_SHA,
                "load_s": round(load_s, 2),
                "resident_gib": round(resident, 2),
                "peak_gib": round(peak, 2),
                "chunk_ms": round(chunk_s * 1000, 1),
                "num_frames": len(frames),
                "action_shape": list(shape),
                **POLICY,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run stage 4 (downloads ~33 GB)")
    parser.add_argument("--no-install", action="store_true", help="skip the diffusers pip install")
    args = parser.parse_args()

    print(f"Cosmos 3 ROCm probe -- {REPO}", flush=True)
    print(f"python {sys.version.split()[0]} @ {sys.executable}", flush=True)
    print(f"HF_HOME={os.environ.get('HF_HOME', '<unset>')}", flush=True)
    print("-" * 72, flush=True)

    stage0_torch()
    stage1_diffusers(install=not args.no_install)
    stage2_module_tree()
    stage3_construct_from_config()

    if not args.full:
        print("-" * 72, flush=True)
        print("Cheap gate passed. The ROCm story survives structural inspection.", flush=True)
        print("Re-run with --full to download weights and time a real policy call.", flush=True)
        return

    stage4_real_call()


if __name__ == "__main__":
    main()
