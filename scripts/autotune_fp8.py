"""Autotune the FP8 Triton flash-attention kernel at the Cosmos production shape.

Drives the persistent JSON cache at ``~/.cache/repercep/fp8_autotune.json``
(overridable via ``REPERCEP_FP8_AUTOTUNE_CACHE``). The kernel itself is
``@triton.autotune``-decorated; this script forces a search at the shapes
we care about and reports the winner.

Two modes:

- *autotune mode* (default): trigger the kernel's built-in Triton autotuner
  over its 19-config grid; the winner is cached on disk.
- *manual mode* (``--manual``): bench a hand-picked set of (BLOCK_M, BLOCK_N,
  num_warps, num_stages) tuples on the same shape and pick the median-best.
  More robust under GPU contention because we control warmup/rep depth.

Usage::

    sg render -c "sg video -c '.venv/bin/python scripts/autotune_fp8.py'"
    .venv/bin/python scripts/autotune_fp8.py --shapes 8192 32768 109120
    .venv/bin/python scripts/autotune_fp8.py --rerun  # clear cache + retune
    .venv/bin/python scripts/autotune_fp8.py --manual --shapes 109120

Prints a markdown table comparing:
- ``fixed-fp8``     — pre-autotune fallback config (Session-9 sweet spot)
- ``autotuned-fp8`` — tuned config picked by the autotuner / manual search
- ``sdpa``          — torch SDPA (= aotriton flash on ROCm)

at each shape. The production Cosmos call is B=2 H=32 D=128 S=109120.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path


def _setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for sub in ("src", "kernels"):
        path = repo_root / sub
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_setup_imports()


import torch  # noqa: E402


def _bench(fn, q, k, v, *, warmup=10, iters=10) -> float:
    """Mean wall time per call in seconds (CUDA-synced).

    Warmup is 10 calls — the autotune-cached path needs to recompile the
    fixed-config kernel on the first call after a cache hit, then takes 2-3
    more calls to reach steady state under GPU contention.
    """
    try:
        for _ in range(warmup):
            fn(q, k, v)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(q, k, v)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        torch.cuda.empty_cache()
        return float("nan")


def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


# Hand-picked manual-search grid — the configs we believe matter most for the
# Cosmos production shape (B=2 H=32 D=128 S>=8k). Order matters: the first
# config that succeeds without OOM and survives 10 warmup iterations is
# treated as the canonical baseline; we then score the rest by relative speed.
_MANUAL_CONFIGS = (
    {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 256, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 256, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 256, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 256, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 256, "BLOCK_N": 128, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2},
)


def _bench_one_config(
    cfg: dict,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> float:
    """Bench a specific (BLOCK_M, BLOCK_N, num_warps, num_stages) config.

    Writes ``cfg`` to a private cache file, invokes the kernel via the cached
    path, then restores the cache.  This isolates per-config benches from one
    another without relying on Triton's autotune state.
    """
    from triton_kernels.fp8_flash_attn import (
        _cache_key,
        _cache_load,
        _cache_save,
        fp8_flash_attention,
    )

    b, h, sq, d = q.shape
    skv = k.shape[2]
    causal = False

    # Snapshot the cache, install this one config, run, restore.
    saved = _cache_load()
    forced = dict(saved)
    forced[_cache_key(b, h, sq, skv, d, causal)] = cfg
    _cache_save(forced)
    try:
        return _bench(fp8_flash_attention, q, k, v, warmup=warmup, iters=iters)
    finally:
        _cache_save(saved)


def _manual_autotune(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> tuple[dict, float, list[tuple[dict, float]]]:
    """Bench every config in ``_MANUAL_CONFIGS``; return the winner.

    Returns (best_cfg, best_time, all_results).  ``best_time`` is in seconds.
    """
    from triton_kernels.fp8_flash_attn import (
        _cache_key,
        _cache_load,
        _cache_save,
    )

    results: list[tuple[dict, float]] = []
    for cfg in _MANUAL_CONFIGS:
        t = _bench_one_config(cfg, q, k, v, warmup=warmup, iters=iters)
        results.append((cfg, t))
        cfg_str = (
            f"M={cfg['BLOCK_M']:>3} N={cfg['BLOCK_N']:>3} "
            f"w={cfg['num_warps']} s={cfg['num_stages']}"
        )
        if math.isnan(t):
            print(f"  [manual] {cfg_str}:  n/a (failed)", file=sys.stderr, flush=True)
        else:
            print(
                f"  [manual] {cfg_str}:  {t * 1000:7.2f} ms",
                file=sys.stderr,
                flush=True,
            )

    valid = [(c, t) for c, t in results if not math.isnan(t)]
    if not valid:
        raise RuntimeError("All configs failed in manual autotune.")
    valid.sort(key=lambda x: x[1])
    best_cfg, best_t = valid[0]

    # Persist the winner.
    b, h, sq, d = q.shape
    skv = k.shape[2]
    causal = False
    cache = _cache_load()
    cache[_cache_key(b, h, sq, skv, d, causal)] = best_cfg
    _cache_save(cache)

    return best_cfg, best_t, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--shapes",
        type=int,
        nargs="+",
        default=[8192, 32768, 65536, 109120],
        help="Sequence lengths to autotune and benchmark",
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Delete the cache before tuning (forces a fresh search)",
    )
    parser.add_argument(
        "--cache",
        default=str(Path.home() / ".cache" / "repercep" / "fp8_autotune.json"),
        help="Path to the persistent autotune cache",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help=(
            "Skip the autotune search — assume the cache is already populated "
            "for the requested shapes (useful for validation runs)."
        ),
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Use the hand-picked _MANUAL_CONFIGS grid with our own bench "
            "(warmup=10, iters=N) instead of Triton's autotune. More robust "
            "under GPU contention because we control measurement depth."
        ),
    )
    args = parser.parse_args()

    os.environ["REPERCEP_FP8_AUTOTUNE_CACHE"] = args.cache
    cache_path = Path(args.cache)

    if not torch.cuda.is_available():
        print("No CUDA/ROCm GPU available.", file=sys.stderr)
        return 1

    if args.rerun and cache_path.exists():
        cache_path.unlink()
        print(f"[autotune] cleared cache at {cache_path}")

    # Import after env var is set so the kernel sees the right cache.
    from triton_kernels.fp8_flash_attn import fp8_flash_attention

    device = torch.device("cuda", 0)
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"Torch:   {torch.__version__}")
    print(f"Cache:   {cache_path}")
    print(f"Mode:    {'manual' if args.manual else 'triton.autotune'}")
    print()
    print(f"Shape:   B={args.batch} H={args.heads} D={args.head_dim}")
    print()
    print(
        "| seq_len |    sdpa    | fixed-fp8 | autotuned-fp8 | tuned vs fixed | "
        "tuned vs sdpa | winning config |"
    )
    print("|--:|--:|--:|--:|--:|--:|--|")

    for seq_len in args.shapes:
        torch.manual_seed(0)
        q = (
            torch.randn(
                args.batch, args.heads, seq_len, args.head_dim,
                device=device, dtype=torch.bfloat16,
            )
            / 8.0
        )
        k = (
            torch.randn(
                args.batch, args.heads, seq_len, args.head_dim,
                device=device, dtype=torch.bfloat16,
            )
            / 8.0
        )
        v = torch.randn(
            args.batch, args.heads, seq_len, args.head_dim,
            device=device, dtype=torch.bfloat16,
        )

        # 1. SDPA reference.
        t_sdpa = _bench(_sdpa, q, k, v, iters=args.iters)

        # 2. Fixed-config FP8 (pre-autotune sweet spot).
        os.environ["REPERCEP_FP8_DISABLE_AUTOTUNE"] = "1"
        t_fixed = _bench(fp8_flash_attention, q, k, v, iters=args.iters)
        os.environ.pop("REPERCEP_FP8_DISABLE_AUTOTUNE", None)

        # 3. Autotuned config.
        if args.manual and not args.no_tune:
            print(
                f"  [autotune] manual search at B={args.batch} H={args.heads} "
                f"D={args.head_dim} S={seq_len} ...",
                file=sys.stderr,
                flush=True,
            )
            t_autotune_start = time.perf_counter()
            best_cfg, _best_t, _all_results = _manual_autotune(
                q, k, v, warmup=5, iters=5
            )
            torch.cuda.synchronize()
            autotune_search_s = time.perf_counter() - t_autotune_start
            print(
                f"  [autotune]   ... manual search took {autotune_search_s:.1f}s, "
                f"winner {best_cfg}",
                file=sys.stderr,
                flush=True,
            )
            # The disk cache now holds best_cfg; bench it cleanly.
            t_tuned = _bench(fp8_flash_attention, q, k, v, iters=args.iters)
        elif not args.no_tune:
            print(
                f"  [autotune] triton.autotune at B={args.batch} H={args.heads} "
                f"D={args.head_dim} S={seq_len} ...",
                file=sys.stderr,
                flush=True,
            )
            t_autotune_start = time.perf_counter()
            # Force at least one launch through the autotune path.  Subsequent
            # calls within this process hit the in-memory cache.
            _ = fp8_flash_attention(q, k, v)
            torch.cuda.synchronize()
            autotune_search_s = time.perf_counter() - t_autotune_start
            print(
                f"  [autotune]   ... search took {autotune_search_s:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            t_tuned = _bench(fp8_flash_attention, q, k, v, iters=args.iters)
        else:
            print(
                f"  [autotune] --no-tune: assuming cache has B={args.batch} "
                f"H={args.heads} S={seq_len}",
                file=sys.stderr,
                flush=True,
            )
            t_tuned = _bench(fp8_flash_attention, q, k, v, iters=args.iters)

        # Read back the winning config for this shape.
        import json as _json
        try:
            with cache_path.open("r") as fh:
                cache = _json.load(fh)
        except (OSError, _json.JSONDecodeError):
            cache = {}
        key = (
            f"B{args.batch}_H{args.heads}_Sq{seq_len}_"
            f"Skv{seq_len}_D{args.head_dim}_C0"
        )
        winner = cache.get(key, {})
        winner_str = (
            f"M={winner.get('BLOCK_M')} N={winner.get('BLOCK_N')} "
            f"w={winner.get('num_warps')} s={winner.get('num_stages')}"
            if winner
            else "(not cached)"
        )

        def _ratio(a: float, b: float) -> str:
            if math.isnan(a) or math.isnan(b) or b == 0:
                return "n/a"
            return f"{b / a:.2f}x"

        cells = [
            f"{seq_len}",
            f"{t_sdpa * 1000:7.2f}ms" if not math.isnan(t_sdpa) else "n/a",
            f"{t_fixed * 1000:7.2f}ms" if not math.isnan(t_fixed) else "n/a",
            f"{t_tuned * 1000:7.2f}ms" if not math.isnan(t_tuned) else "n/a",
            _ratio(t_tuned, t_fixed),
            _ratio(t_tuned, t_sdpa),
            winner_str,
        ]
        print("| " + " | ".join(cells) + " |", flush=True)

    print()
    print("Notes:")
    print("- 'fixed-fp8' bypasses the cache (REPERCEP_FP8_DISABLE_AUTOTUNE=1).")
    print("- 'autotuned-fp8' reads from the persistent JSON cache; first call")
    print("  for a given shape runs the autotune search, subsequent calls reuse.")
    print("- '> 1.00x vs sdpa' means FP8 wins.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
