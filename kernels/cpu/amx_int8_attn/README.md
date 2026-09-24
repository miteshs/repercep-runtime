# `amx_int8_attn` — Intel AMX INT8 flash-attention for Repercep

INT8 sibling of `kernels/cpu/amx_attn/` (BF16).  Implements a FlashAttention-2
forward pass using Intel AMX INT8 tile instructions (`TDPBSSD`) plus AVX-512
BF16/FP32 for the online softmax, dynamic quantization, and output rescale.

## What this is

A single PyBind11 C++ extension (`_native`) exporting:

```python
amx_int8_attn._native.flash_attn_int8(
    Q, K, V,         # (B, H, S, D) bf16, contiguous, CPU
    sm_scale: float, # usually 1 / sqrt(D)
    is_causal: bool,
) -> Tensor          # (B, H, S, D) bf16
```

Shapes supported: `D in {64, 128}`, `S` a positive multiple of 32, `B >= 1`,
`H >= 1`.  Q/K/V must share shape, dtype (`bf16`), and contiguity.

## BF16-in / BF16-out contract

The surface mirrors the BF16 sibling exactly — the caller hands the kernel
BF16 Q/K/V and gets BF16 output back.  Quantization to INT8 is dynamic and
happens **inside** the kernel, on a per-tile basis:

* `Q`: per-row symmetric INT8 with FP32 scale (one scale per query token).
  Quantized once per Q-tile.
* `K`: per-row symmetric INT8 with FP32 scale (one scale per key token).
  Quantized per K-tile.
* `V`: per-tile symmetric INT8 with one FP32 scalar scale shared across all
  N_KV * D entries of the tile.  Per-row V quant would put a non-constant
  factor inside the `P @ V` sum and defeat the INT8 fast path; the per-tile
  scale absorbs cleanly into the output rescale.
* `P`: per-row symmetric INT8 with FP32 scale, recomputed every K-tile
  because the online-softmax updates the row magnitudes.

Score-tile dequant uses `q_scale[i] * k_scale[j]`; output dequant uses
`p_scale[i] * v_tile_scale`.  The online softmax and the final
`O /= rowsum` are in FP32 (same as BF16).

## Hardware / OS requirements

* Intel **Sapphire Rapids** (or newer) CPU.  `amx_int8` must appear in
  `/proc/cpuinfo` flags — `setup.py` aborts otherwise.
* Linux **>= 5.16** (for `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`).
* gcc **>= 12** (Ubuntu 24.04 ships 13, which works out-of-the-box).
* PyTorch with the matching C++ ABI.

## Build

From this directory:

```bash
python setup.py build_ext --inplace
```

That produces `_native.cpython-*.so` alongside `__init__.py`.  Or use the
top-level `make kernels-cpu-int8` target.

## Python wrapper

The user-facing API lives at:

```
src/repercep/attention/amx_int8_flash.py
```

That module imports `_native`, probes `/proc/cpuinfo` for the `amx_int8`
feature flag, and exposes the same `available` / `supports` / `__call__`
surface as the BF16 sibling.

## Throughput note

TDPBSSD does INT8 · INT8 → INT32 at **2x** the throughput of TDPBF16PS on
Sapphire Rapids, but per-call dynamic activation quantization adds a fixed
overhead per K-tile.  At short context (S < 1024) the dynamic-quant cost
dominates; the speedup over BF16 is realised at long context.  Treat this
kernel as correctness-first until benchmarked on the target Cosmos shape.

## Tile layout differences vs the BF16 sibling

| Tile      | BF16 (`amx_attn`)        | INT8 (here)              |
|-----------|--------------------------|--------------------------|
| `TMM_Q`   | 16 x 32 bf16             | 16 x 64 int8             |
| `TMM_K`   | 16 x 32 bf16             | 16 x 64 int8             |
| `TMM_S`   | 16 x 16 fp32             | 16 x 16 int32            |
| `TMM_P`   | 16 x 32 bf16             | 16 x 64 int8             |
| `TMM_V`   | 16 x 32 bf16             | 16 x 64 int8             |
| `TMM_O`   | 16 x 16 fp32             | 16 x 16 int32            |

INT8 packs 64 lanes per tile row vs BF16's 32, halving the number of
K-axis chunks needed at the same head_dim.

## Notes

* `_tile_release()` is called at the end of each parallel region so
  oneDNN / IPEX layers that touch AMX after this kernel returns see a
  clean tile-config slot.
* Per-thread scratch buffers (Q_i8, Kt_i8, V_i8, V_amx, P_i8, P_amx,
  S_scratch, O_scratch, O_acc) live on the stack; total upper bound at
  D=128 is ~55 KiB, comfortably inside the per-core L2 on SPR.
* P @ V is dispatched with K-axis zero-padding from 32 to 64 (one
  TDPBSSD chunk) so that the matmul fits a single tile call per output
  half.  See the in-source comment on the PV step for the rationale.
