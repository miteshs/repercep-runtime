"""Benchmark FP8 / FA-3 attention paths on NVIDIA Hopper (sm_90a).

Sibling of ``scripts/bench_fp8.py`` (which targets AMD CDNA3).  The Cosmos
production shape on H100 is B=2, H=32, head_dim=128 and sequence lengths
from 4k tokens (49-frame configs) up to ~109k tokens (the 121-frame
reference config).  This script measures attention-only forward time
across that range for the Hopper ops:

- naive-sdpa            (BF16 SDPA -> cuDNN-FA3 on Hopper, current floor)
- hopper-flash          (the Repercep HopperFlashAttention wrapper:
                         FA-3 if installed, else FA-2, else SDPA)
- fp8-hopper-triton     (the Repercep FP8 Hopper Triton kernel — the
                         autotuned WGMMA kernel from kernels/triton_kernels/
                         fp8_flash_attn_hopper.py.  F29 lives here.)
- transformer-engine    (the optional TE op; only if `transformer_engine`
                         imports — see F28 in BUILD_LOG for the cu12/cu13
                         install hazard.)

Usage::

    .venv/bin/python scripts/bench_fp8_hopper.py
    .venv/bin/python scripts/bench_fp8_hopper.py --shapes 8192 16384 32768

Output is a markdown table.  Cosmos's 121-f spatial attention runs at
S~109k; extrapolate from the largest practical S here.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path


def _setup_imports() -> None:
    """Add src/ and kernels/ to sys.path so the script runs from a worktree."""
    repo_root = Path(__file__).resolve().parents[1]
    for sub in ("src", "kernels"):
        path = repo_root / sub
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_setup_imports()


import torch  # noqa: E402

from repercep.attention.fp8_hopper_triton import FP8HopperTritonAttention  # noqa: E402
from repercep.attention.hopper_flash import HopperFlashAttention  # noqa: E402
from repercep.attention.naive import NaiveAttention  # noqa: E402

try:
    from repercep.attention.transformer_engine import TransformerEngineAttention

    _HAS_TE = True
except Exception:
    _HAS_TE = False


def _bench_one(
    op: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Mean wall time per call in seconds, CUDA-synced.  NaN on failure."""
    try:
        for _ in range(warmup):
            _ = op(q, k, v)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = op(q, k, v)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")
    except Exception as e:  # pragma: no cover — defensive
        print(f"  {getattr(op, 'name', op.__class__.__name__)}: FAILED ({type(e).__name__}: {e})",
              file=sys.stderr)
        torch.cuda.empty_cache()
        return float("nan")


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (out.float() - ref.float()).abs()
    scale = ref.float().abs().mean().item() + 1e-6
    return float(diff.mean().item() / scale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to sweep",
    )
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU available — bench requires an H100 (or any sm_90+).")
        return 1

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}  (sm_{props.major}{props.minor})")
    print(f"Torch:  {torch.__version__}")
    if props.major < 9:
        print("WARNING: this is the Hopper bench; running on pre-sm_90 silicon "
              "means FP8 paths may be unavailable.")
    print()

    ops: dict[str, Callable] = {
        "naive-sdpa": NaiveAttention(),
        "hopper-flash": HopperFlashAttention(),
        "fp8-hopper-triton": FP8HopperTritonAttention(),
    }
    if _HAS_TE:
        try:
            ops["transformer-engine"] = TransformerEngineAttention()
        except Exception as e:
            print(f"  transformer-engine: ctor failed ({type(e).__name__}: {e})")
    else:
        print("  transformer-engine: not installed (F28 install fix pending — skipped)")

    for name, op in ops.items():
        avail = getattr(op, "available", True)
        print(f"  {name:20s}: available={avail}")
    print()

    cols = list(ops.keys())
    print(f"## Cosmos-DiT-shaped attention on Hopper: "
          f"B={args.batch} H={args.heads} D={args.head_dim}")
    print()
    # Per-op rel_err (vs SDPA reference).  Previously this column showed
    # min across ops, which silently hid FP8's quantization error behind
    # any BF16 op that happened to match SDPA bit-exactly (HopperFlash with
    # FA-2 does, since BF16 reductions on Hopper give identical results).
    err_cols = [f"err({c})" for c in cols if c != "naive-sdpa"]
    header = "| seq_len | " + " | ".join(cols) + " | best vs SDPA | " + " | ".join(err_cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 2 + len(err_cols))) + "|"
    print(header)
    print(sep)

    for seq_len in args.shapes:
        torch.manual_seed(0)
        q = torch.randn(args.batch, args.heads, seq_len, args.head_dim,
                        device=device, dtype=torch.bfloat16) / 8.0
        k = torch.randn(args.batch, args.heads, seq_len, args.head_dim,
                        device=device, dtype=torch.bfloat16) / 8.0
        v = torch.randn(args.batch, args.heads, seq_len, args.head_dim,
                        device=device, dtype=torch.bfloat16)

        ref_out = NaiveAttention()(q, k, v)

        timings: dict[str, float] = {}
        errors: dict[str, float] = {}
        for name, op in ops.items():
            t = _bench_one(op, q, k, v, iters=args.iters)
            timings[name] = t
            if math.isnan(t):
                errors[name] = float("nan")
            elif name != "naive-sdpa":
                with torch.no_grad():
                    try:
                        out = op(q, k, v)
                        errors[name] = _rel_err(out, ref_out)
                    except Exception:
                        errors[name] = float("nan")
            else:
                errors[name] = 0.0

        cells: list[str] = [f"{seq_len:5d}"]
        for name in cols:
            t = timings[name]
            cells.append("n/a" if math.isnan(t) else f"{t * 1000:7.2f}ms")
        sdpa_t = timings["naive-sdpa"]
        best_fp8 = min(
            (t for n, t in timings.items() if n != "naive-sdpa" and not math.isnan(t)),
            default=float("inf"),
        )
        speedup = (
            f"{sdpa_t / best_fp8:.2f}x"
            if not math.isinf(best_fp8) and sdpa_t > 0
            else "n/a"
        )
        cells.append(speedup)
        for name in cols:
            if name == "naive-sdpa":
                continue
            e = errors[name]
            cells.append("n/a" if math.isnan(e) else f"{e:.4f}")
        print("| " + " | ".join(cells) + " |")

    print()
    print("Notes:")
    print("- naive-sdpa on Hopper goes through cuDNN's FA-3 backend in torch 2.8+.")
    print("- hopper-flash prefers FA-3 if installed (PYPI wheel or built from source);")
    print("  otherwise FA-2; otherwise SDPA.  Check the wrapper for which path lit up.")
    print("- fp8-hopper-triton autotunes the WGMMA tile per shape on first call; the")
    print("  first row may include autotune time.  Re-run to see steady-state.")
    print("- rel_err is mean(|out - ref|) / mean(|ref|).")
    print("- For the Cosmos 121-f spatial attention S~109k; extrapolate from S=32768.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
