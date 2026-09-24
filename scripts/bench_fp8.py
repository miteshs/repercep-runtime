"""Benchmark FP8 attention paths against SDPA on Cosmos-DiT-like shapes.

The Cosmos-Predict-7B DiT runs attention with B=2, H=32, head_dim=128 and
sequence lengths from 4k tokens (49-frame configs) up to ~109k tokens (the
121-frame reference config).  This script measures attention-only forward
time across that range for the four ops:

- naive-sdpa (SDPA -> aotriton flash, the current production path)
- rocm-ck-flash (CK flash-attn wrapper -- inert without the CK build)
- fp8-scaled-mm (per-head FP8 GEMMs via torch._scaled_grouped_mm)
- fp8-triton-flash (fused FP8 flash-attention Triton kernel)

Usage:
    .venv/bin/python scripts/bench_fp8.py
    .venv/bin/python scripts/bench_fp8.py --shapes 8192 16384 32768

Output is a markdown table; pipe into docs/OPTIMIZATION.md or stdout.
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

from repercep.attention.fp8_scaled_mm import FP8ScaledMMAttention  # noqa: E402
from repercep.attention.fp8_triton import FP8TritonAttention  # noqa: E402
from repercep.attention.naive import NaiveAttention  # noqa: E402


def _bench_one(
    op: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Returns mean wall time per call in seconds.  CUDA-synced.

    Returns ``float('nan')`` if the op raises (e.g. unsupported shape, OOM).
    """
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
    except Exception as e:  # pragma: no cover - defensive
        print(f"  {op.name}: FAILED ({type(e).__name__}: {e})", file=sys.stderr)
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
        print("No CUDA/ROCm GPU available -- bench requires the MI300X.")
        return 1

    device = torch.device("cuda", 0)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Torch:  {torch.__version__}")
    print()

    ops = {
        "naive-sdpa": NaiveAttention(),
        "fp8-scaled-mm": FP8ScaledMMAttention(),
        "fp8-triton-flash": FP8TritonAttention(),
    }
    for name, op in ops.items():
        avail = getattr(op, "available", True)
        print(f"  {name:18s}: available={avail}")
    print()

    # Print header.
    cols = list(ops.keys())
    print(f"## Cosmos-DiT-shaped attention: B={args.batch} H={args.heads} D={args.head_dim}")
    print()
    header = "| seq_len | " + " | ".join(cols) + " | best vs SDPA | rel_err (FP8) |"
    sep = "|" + "|".join(["---"] * (len(cols) + 3)) + "|"
    print(header)
    print(sep)

    for seq_len in args.shapes:
        torch.manual_seed(0)
        # Synthetic but realistic-magnitude tensors.  /8.0 keeps Q/K small
        # enough that exp(qk * scale) doesn't saturate the BF16 range.
        q = (
            torch.randn(
                args.batch, args.heads, seq_len, args.head_dim, device=device, dtype=torch.bfloat16
            )
            / 8.0
        )
        k = (
            torch.randn(
                args.batch, args.heads, seq_len, args.head_dim, device=device, dtype=torch.bfloat16
            )
            / 8.0
        )
        v = torch.randn(
            args.batch, args.heads, seq_len, args.head_dim, device=device, dtype=torch.bfloat16
        )

        # SDPA reference output for error reporting.
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
                    out = op(q, k, v)
                errors[name] = _rel_err(out, ref_out)
            else:
                errors[name] = 0.0

        cells: list[str] = [f"{seq_len:5d}"]
        for name in cols:
            t = timings[name]
            if math.isnan(t):
                cells.append("n/a")
            else:
                cells.append(f"{t * 1000:7.2f}ms")
        sdpa_t = timings["naive-sdpa"]
        best_fp8 = min(
            (t for n, t in timings.items() if n != "naive-sdpa" and not math.isnan(t)),
            default=float("inf"),
        )
        if not math.isinf(best_fp8) and sdpa_t > 0:
            cells.append(f"{sdpa_t / best_fp8:.2f}x")
        else:
            cells.append("n/a")
        # Best FP8 error.
        fp8_errs = [e for n, e in errors.items() if n != "naive-sdpa" and not math.isnan(e)]
        if fp8_errs:
            cells.append(f"{min(fp8_errs):.3f}")
        else:
            cells.append("n/a")

        print("| " + " | ".join(cells) + " |")

    print()
    print("Notes:")
    print("- naive-sdpa uses torch SDPA, which routes through aotriton flash on ROCm.")
    print("- rel_err is mean(|out - ref|) / mean(|ref|), the FP8 path's lowest err.")
    print("- 'best vs SDPA' = SDPA_time / fastest_FP8_time; >1 means FP8 wins.")
    print("- For Cosmos's full 121-f config the spatial attention runs at S~109k;")
    print("  extrapolate from S=32768 (largest practical here without OOM warning).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
