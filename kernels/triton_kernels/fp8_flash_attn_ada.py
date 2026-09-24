"""FP8 flash-attention Triton kernel for Ada Lovelace (sm_89 / RTX 4090, RTX 2000 Ada, L40S).

This is the *kernel-layer* code — no abstractions, no Protocols.  The thin
loader in ``src/repercep/attention/fp8_ada_triton.py`` wraps it as an
``AttentionOp``.

Algorithm: standard FlashAttention-2 (Dao 2023) — tile Q over the
sequence axis, stream K/V tiles through, maintain a running max + running
sum + running accumulator (online softmax), never materialize the S^2
scores matrix.  The matmuls are done in FP8 (e4m3fn — IEEE, with inf/nan)
via ``tl.dot`` with explicit dtype casts; the running statistics and
accumulator stay in FP32 for accuracy.  This is the Ada sibling of
``fp8_flash_attn_hopper.py`` (sm_90a) and ``fp8_flash_attn.py``
(gfx942).  Algorithm is identical to the Hopper kernel; the divergences
are the underlying tensor-core instruction (Ada uses sm_89's
``mma.sync.aligned.m16n8k32.f32.e4m3.e4m3`` from PTX 8.0+ rather than
Hopper's warpgroup ``wgmma.mma_async``) and a more conservative autotune
grid (Ada's 100 KiB SMEM/block sits between Hopper's 228 KiB and
CDNA3's 64 KiB).

Why this works on sm_89 at all.  Ada Lovelace introduced FP8 ISA support
(E4M3 + E5M2) in its 4th-gen tensor cores — the same dtypes Hopper has,
but exposed through the older synchronous ``mma.sync`` family instead of
Hopper's asynchronous warpgroup ``wgmma``.  Crucially, Triton's NVPTX
backend dispatches ``tl.dot`` with FP8 operands to ``mma.sync`` on sm_89
and to ``wgmma`` on sm_90a — same Triton source, different PTX.  As long
as we avoid Hopper-only intrinsics (TMA descriptors, ``wgmma`` fence
ops, ``tl.async_copy``), the kernel "just compiles" on Ada.  The gfx942
kernel is structurally closer to what Ada wants than the Hopper kernel
is, because AMD doesn't have async warpgroup matmuls either; we model
the Ada kernel after the Hopper version though because both share the
e4m3fn dtype (vs gfx942's e4m3fnuz).

Autotune.  The kernel exposes a tile shape (``BLOCK_M``, ``BLOCK_N``)
and launch-time meta (``num_warps``, ``num_stages``).  Ada Lovelace has
100 KiB SMEM/block — enough for 128x128 and 128x256 tiles at
``num_warps=4`` or 8, but ``256x256`` does not fit comfortably alongside
the FP32 accumulator.  Ada's SMs are narrower than Hopper's (4 SM
partitions vs Hopper's wider organization), so ``num_warps=4`` is often
the sweet spot rather than 8.  We sweep ``num_stages`` in (2, 3) — the
synchronous ``mma.sync`` pipeline doesn't benefit from deep software
pipelining the way ``wgmma`` does.

Persistent JSON cache is keyed on shape; cache file is distinct from the
Hopper and CDNA3 files because the optimal configs differ across
architectures.

References:
- Dao, FlashAttention-2: Faster Attention with Better Parallelism (2023)
- NVIDIA PTX ISA 8.0 §9.7.13.5 (mma.sync FP8 variants) for the underlying
  sm_89 matmul.
- NVIDIA Ada Lovelace Architecture Whitepaper §3.3 (4th-gen tensor cores).
"""

# Triton kernels are eval-loaded; this file is exempt from mypy/ruff
# (see kernels/README.md).
# ruff: noqa
# type: ignore

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


# FP8 max for e4m3fn on Ada (same as Hopper) — IEEE-style with inf/nan,
# finite max is 448.  Unlike gfx942's e4m3fnuz (max 240), the NVIDIA FP8
# format is full IEEE-style.
FP8_E4M3_MAX = 448.0

# Persistent shape→config cache.  One autotune per (B, H, Sq, Skv, D, causal)
# per host; the winner is recorded on disk and reused across processes.  Path
# is overridable via REPERCEP_FP8_AUTOTUNE_CACHE_ADA so tests can pin a tmp
# file.  Distinct from the Hopper and CDNA3 cache files because the optimal
# tile shapes differ across architectures.
_DEFAULT_CACHE_PATH = Path.home() / ".cache" / "repercep" / "fp8_autotune_ada.json"


def _cache_path() -> Path:
    override = os.environ.get("REPERCEP_FP8_AUTOTUNE_CACHE_ADA")
    if override:
        return Path(override)
    return _DEFAULT_CACHE_PATH


def _cache_load() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        with path.open("r") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _cache_save(cache: dict) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
    tmp.replace(path)


def _cache_key(B: int, H: int, Sq: int, Skv: int, D: int, causal: bool) -> str:
    """Stable string key for the JSON cache.

    Batch enters the key because B affects the program-grid columns: more
    columns means more parallelism, which can change the optimal tile.
    """
    return f"B{B}_H{H}_Sq{Sq}_Skv{Skv}_D{D}_C{int(causal)}"


# Autotune search grid.  Constraints baked in for Ada Lovelace (sm_89):
# - BLOCK_M, BLOCK_N >= 32 (mma.sync FP8 tile floor on sm_89:
#   m16n8k32 is the FP8 mma; tiling for the kernel sees larger blocks).
# - BLOCK_N <= BLOCK_M*2 (avoid pathological SMEM layouts).
# - Ada has 100 KiB SMEM/block (vs Hopper's 228 KiB, vs CDNA3's 64 KiB),
#   so 256x256 does not fit alongside the FP32 accumulator; we cap at
#   192x192 effective and let Triton's compiler filter overflows.
# - num_warps=4 is often the sweet spot on Ada because the SMs are
#   narrower than Hopper's.  num_warps=8 explored but only for the
#   largest tiles where there's actual work to amortize.
# - num_stages: Ada's mma.sync (synchronous) doesn't benefit from deep
#   software pipelining the way Hopper's wgmma does.  Sweep (2, 3).
#
# Total grid is ~16 configs; at the test shape (S=4096) each launch is
# ~10-30 ms so the tune tax is ~0.5-1 s — small enough for a single-shot
# correctness test to not feel painful.
def _autotune_configs() -> list:
    configs: list = []
    # Tile shape candidates, ordered roughly by expected goodness for
    # FP8 flash attention on Ada Lovelace.  192x192 is the largest tile
    # that fits comfortably; 256x256 overflows SMEM with the FP32 acc.
    tile_shapes = (
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (128, 256),
        (256, 64),
        (256, 128),
    )
    warp_choices = (4, 8)
    stage_choices = (2, 3)
    for bm, bn in tile_shapes:
        # mma.sync correctness floor.
        if bm < 32 or bn < 32:
            continue
        # 100 KiB SMEM ceiling: filter anything that obviously overflows.
        # FP8 inputs (1B each) + FP32 acc (4B) + softmax stats; 256x128 is
        # roughly the limit.  Triton's compiler filters the rest at
        # autotune time.
        if bm * bn > 256 * 128:
            continue
        for nw in warp_choices:
            # num_warps=8 needs enough work to amortize; skip for the
            # smallest tile.
            if nw == 8 and bm * bn < 128 * 64:
                continue
            for ns in stage_choices:
                configs.append(
                    triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn},
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return configs


_AUTOTUNE_CONFIGS = _autotune_configs()


@triton.jit
def _fp8_flash_attn_fwd_impl(
    Q,
    K,
    V,
    sm_scale,  # 1/sqrt(d), float
    qs,
    ks,
    vs,  # per-tensor dequant scales (B*H,) float32
    Out,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    B,
    H,
    Sq,
    Skv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """One program = one (B, H, Q-tile) triple.

    Q and Out tiles: (BLOCK_M, BLOCK_D).
    K/V tiles streamed: (BLOCK_N, BLOCK_D).

    The accumulator and softmax statistics are FP32; the matmuls are FP8
    via Triton's ``tl.dot`` which lowers to ``mma.sync`` FP8 instructions
    on sm_89 (Ada Lovelace) and ``wgmma`` on sm_90a (Hopper).
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    # Per-(B,H) dequant scales.  q_scale * k_scale enters the QK^T result;
    # p_scale * v_scale enters the PV result.  We carry them as plain floats.
    q_scale = tl.load(qs + pid_bh)
    k_scale = tl.load(ks + pid_bh)
    v_scale = tl.load(vs + pid_bh)

    # Offset Q, K, V, Out to this (b, h) slice.
    q_off = b * stride_qb + h * stride_qh
    k_off = b * stride_kb + h * stride_kh
    v_off = b * stride_vb + h * stride_vh
    o_off = b * stride_ob + h * stride_oh

    # Row indices into Q for this program (m-tile).
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load this Q tile.  Q stays resident in SRAM/registers for the whole
    # inner loop.  Shape: (BLOCK_M, BLOCK_D), FP8.
    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < Sq
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)  # FP8 dtype

    # Online softmax state, FP32.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Effective softmax scale: sm_scale * q_scale * k_scale.  This folds the
    # dequant into the temperature in one multiplication per (B,H), instead
    # of per-element after the dot.
    qk_scale = sm_scale * q_scale * k_scale

    # Loop over K/V tiles.
    n_blocks = tl.cdiv(Skv, BLOCK_N)

    for n_idx in range(0, n_blocks):
        n_start = n_idx * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # Load K tile (BLOCK_N, BLOCK_D), FP8.
        k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = offs_n[:, None] < Skv
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # qk = Q @ K^T (BLOCK_M, BLOCK_N), FP32 accumulator.  tl.dot's
        # accumulator dtype is FP32 by default for FP8 inputs; Triton emits
        # ``mma.sync.aligned.m16n8k32.f32.e4m3.e4m3`` on sm_89 via the
        # NVPTX backend.
        qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32)
        qk = qk * qk_scale

        # Apply causal mask if requested.
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Mask out-of-range KV positions (last tile partial).
        kv_mask = offs_n[None, :] < Skv
        qk = tl.where(kv_mask, qk, float("-inf"))

        # Online softmax (running max + running sum).
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)  # rescale factor for prior acc
        p = tl.exp(qk - m_new[:, None])  # (M, N), FP32, in [0, 1]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Rescale the running accumulator before adding new contribution.
        acc = acc * alpha[:, None]

        # Load V tile (BLOCK_N, BLOCK_D), FP8.
        v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = offs_n[:, None] < Skv
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Quantize P to FP8 for the PV matmul.  P is in [0, 1] so a fixed
        # scale (FP8_MAX = 448 for e4m3fn) saturates the FP8 range without
        # per-tile recomputation; the dequant factor is 1/FP8_MAX, folded
        # into v_scale below.
        p_fp8 = (p * FP8_MAX).to(tl.float8e4nv)
        pv_scale = v_scale / FP8_MAX

        # acc += P @ V  (BLOCK_M, BLOCK_D), FP32 accumulator + FP8 inputs.
        pv = tl.dot(p_fp8, v_tile, out_dtype=tl.float32)
        acc = acc + pv * pv_scale

        m_i = m_new

    # Normalize.
    acc = acc / l_i[:, None]

    # Store the output (cast back to the input dtype on the Python side via
    # the Out tensor's dtype — Triton stores match Out's pointer dtype).
    out_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < Sq
    tl.store(out_ptrs, acc, mask=o_mask)


# The autotuned kernel — same body, with @triton.autotune over the search
# grid.  Triton's autotuner re-launches the kernel once per config in the
# grid on the first call for a new ``key`` tuple, picks the fastest, caches
# in-process.  We additionally persist the winner to JSON in
# ``fp8_flash_attention`` below; on subsequent processes the launcher
# bypasses ``_fp8_flash_attn_fwd_autotuned`` and calls the fixed-config
# kernel directly using the cached config.
#
# Key includes (Sq, Skv, BLOCK_D, H, CAUSAL).
def _autotune_bench(kernel_call, **kwargs):  # type: ignore[no-untyped-def]
    return triton.testing.do_bench(kernel_call, warmup=25, rep=50, **kwargs)


_fp8_flash_attn_fwd_autotuned = triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["Sq", "Skv", "BLOCK_D", "H", "CAUSAL"],
    do_bench=_autotune_bench,
)(_fp8_flash_attn_fwd_impl)


def _per_bh_scale(x: torch.Tensor) -> torch.Tensor:
    """Compute one quantization scale per (batch, head) — shape (B*H,).

    Returns the *dequant* scale (i.e. multiplier you apply to the FP8 value
    to recover the original BF16 magnitude).
    """
    # x is (B, H, S, D); amax over last two dims.
    B, H, S, D = x.shape
    amax = x.abs().reshape(B * H, -1).amax(dim=1).clamp(min=1e-6)
    # Quantization scale: we want x * q_scale in [-FP8_MAX, FP8_MAX].
    # Dequant scale (returned) is 1 / q_scale = amax / (0.95 * FP8_MAX).
    return (amax / (0.95 * FP8_E4M3_MAX)).to(torch.float32)


def _quantize(x: torch.Tensor, dequant_scale: torch.Tensor) -> torch.Tensor:
    """Quantize (B, H, S, D) BF16 to FP8 using per-(B,H) dequant scales.

    ``dequant_scale`` is shape (B*H,).  The quantization multiplier is its
    reciprocal.  Target dtype is ``torch.float8_e4m3fn`` (IEEE e4m3 with
    inf/nan), the NVIDIA-native FP8 representation on both Ada and Hopper.
    """
    B, H, S, D = x.shape
    q_mul = (1.0 / dequant_scale).reshape(B, H, 1, 1)
    scaled = (x.to(torch.float32) * q_mul).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn)


# Static fallback config — used if autotune is disabled or fails.  On Ada
# Lovelace the narrower SMs + smaller (relative to Hopper) SMEM budget
# prefer a 128x64 / num_warps=4 tile as a baseline.  Autotune typically
# improves on this for a given shape.
_FALLBACK_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 64,
    "num_warps": 4,
    "num_stages": 2,
}


def _resolve_config(B: int, H: int, Sq: int, Skv: int, D: int, causal: bool) -> dict:
    """Return ``{"BLOCK_M": ..., "BLOCK_N": ..., "num_warps": ..., "num_stages": ...}``
    for this shape, consulting the on-disk JSON cache."""
    cache = _cache_load()
    key = _cache_key(B, H, Sq, Skv, D, causal)
    cached = cache.get(key)
    if isinstance(cached, dict) and all(
        k in cached for k in ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages")
    ):
        return cached
    return {}


def _record_config(B: int, H: int, Sq: int, Skv: int, D: int, causal: bool, cfg: dict) -> None:
    cache = _cache_load()
    cache[_cache_key(B, H, Sq, Skv, D, causal)] = cfg
    _cache_save(cache)


def fp8_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Public Python entry point — fused FP8 flash attention (Ada Lovelace).

    Resolution order for the launch config:

    1. If ``REPERCEP_FP8_DISABLE_AUTOTUNE`` is set, use ``_FALLBACK_CONFIG``
       (the pre-tune tile shape — useful for A/B comparisons).
    2. Otherwise check the persistent JSON cache at
       ``~/.cache/repercep/fp8_autotune_ada.json`` (or
       ``$REPERCEP_FP8_AUTOTUNE_CACHE_ADA``) for a winning config keyed on
       ``(B, H, Sq, Skv, D, causal)``.  Hit: launch the fixed-config
       kernel.  Miss: fall through.
    3. On cache miss: dispatch through the ``@triton.autotune``-decorated
       kernel; Triton picks the best config in-process.  After the call
       we extract the winner from Triton's cache and persist it.

    Args:
        q, k, v: (B, H, S, D) BF16 or FP16 tensors.
        causal: apply a causal mask.
        scale: softmax temperature; default 1/sqrt(D).

    Returns:
        Output (B, H, S, D) in the dtype of ``q``.
    """
    import math

    assert q.shape == k.shape == v.shape, "MHA only (same S); cross-attn TODO"
    B, H, Sq, D = q.shape
    Skv = k.shape[2]
    assert D in (32, 64, 128, 256), f"head_dim {D} not in compiled tile shapes"

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Per-(B,H) dequant scales.
    q_scales = _per_bh_scale(q)
    k_scales = _per_bh_scale(k)
    v_scales = _per_bh_scale(v)

    q_fp8 = _quantize(q, q_scales).contiguous()
    k_fp8 = _quantize(k, k_scales).contiguous()
    v_fp8 = _quantize(v, v_scales).contiguous()

    out = torch.empty_like(q, dtype=torch.float32)

    disable_autotune = os.environ.get("REPERCEP_FP8_DISABLE_AUTOTUNE", "")
    if disable_autotune in ("1", "true", "on"):
        cfg = dict(_FALLBACK_CONFIG)
        autotune_used = False
    else:
        cfg = _resolve_config(B, H, Sq, Skv, D, causal)
        autotune_used = not cfg
        if not cfg:
            cfg = {}  # signal: use the autotuner

    if autotune_used:
        # Cache miss — dispatch through the autotuner.  The grid in this
        # branch must NOT bind BLOCK_M (it's part of the tuned meta).
        grid = lambda meta: (triton.cdiv(Sq, meta["BLOCK_M"]), B * H)
        _fp8_flash_attn_fwd_autotuned[grid](
            q_fp8,
            k_fp8,
            v_fp8,
            scale,
            q_scales,
            k_scales,
            v_scales,
            out,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            q_fp8.stride(3),
            k_fp8.stride(0),
            k_fp8.stride(1),
            k_fp8.stride(2),
            k_fp8.stride(3),
            v_fp8.stride(0),
            v_fp8.stride(1),
            v_fp8.stride(2),
            v_fp8.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            Sq,
            Skv,
            BLOCK_D=D,
            CAUSAL=causal,
            FP8_MAX=FP8_E4M3_MAX,
        )
        # Pull the winning config back out of Triton's autotuner cache and
        # persist it to disk so the next process can skip the search.
        try:
            best = _fp8_flash_attn_fwd_autotuned.best_config
            winner = {
                "BLOCK_M": int(best.kwargs["BLOCK_M"]),
                "BLOCK_N": int(best.kwargs["BLOCK_N"]),
                "num_warps": int(best.num_warps),
                "num_stages": int(best.num_stages),
            }
            _record_config(B, H, Sq, Skv, D, causal, winner)
        except (AttributeError, KeyError):
            # Old triton without best_config exposed — fail silent; the
            # next process will autotune again.
            pass
    else:
        # Cache hit — launch with the fixed config directly.
        grid = (triton.cdiv(Sq, cfg["BLOCK_M"]), B * H)
        _fp8_flash_attn_fwd_impl[grid](
            q_fp8,
            k_fp8,
            v_fp8,
            scale,
            q_scales,
            k_scales,
            v_scales,
            out,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            q_fp8.stride(3),
            k_fp8.stride(0),
            k_fp8.stride(1),
            k_fp8.stride(2),
            k_fp8.stride(3),
            v_fp8.stride(0),
            v_fp8.stride(1),
            v_fp8.stride(2),
            v_fp8.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            Sq,
            Skv,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_D=D,
            CAUSAL=causal,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return out.to(q.dtype)


__all__ = ["fp8_flash_attention"]
