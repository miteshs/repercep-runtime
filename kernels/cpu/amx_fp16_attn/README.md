# `amx_fp16_attn` — Intel AMX FP16 flash-attention for Repercep (Granite Rapids)

Granite Rapids (GNR) sibling of `kernels/cpu/amx_attn/` (which targets
Sapphire/Emerald Rapids via AMX_BF16).  GNR is the first Intel Xeon silicon
that exposes an FP16 tile-matmul instruction (`TDPFP16PS`, exposed by the
`_tile_dpfp16ps` intrinsic), so this kernel is the FP16 fast path for that
generation.  SPR/EMR cannot run it (no `amx_fp16` flag); they continue to
use the BF16 sibling for BF16 inputs and fall through to the AVX-512_FP16
SDPA path for FP16 inputs.

## Status: SCAFFOLD

This directory currently lands the file tree, the refuse-on-non-GNR build
gate, the Python wrapper plumbing, and a compileable C++ stub whose
`flash_attn_fp16` entry point delegates to
`at::scaled_dot_product_attention` so callers see correct numerics while
the AMX inner loops are still being filled in.

The algorithmic skeleton lives in `flash_attn_amx_fp16.cpp` as a sequence
of `// TODO(GNR):` markers covering:

1. `pack_K_for_amx_B_fp16` — K -> `(B, H, D/2, S, 2)` pre-pack (FP16 mirror
   of the BF16 sibling's `pack_K_for_amx_B`).
2. `pack_V_pairs_fp16` — V -> `(B, H, S/2, D, 2)` pre-pack.
3. `process_q_tile_fp16` — the FlashAttention-2 inner loop driven by
   `_tile_dpfp16ps` for both Q@K^T and P@V, with the AVX-512_FP16 cast
   intrinsics on the softmax output (GNR carries the full FP16 ISA so the
   BF16 sibling's bit-shift cast trick is unnecessary here).
4. The OMP-parallel driver inside `flash_attn_fp16` (currently the SDPA
   fallback line marked `TODO(GNR):`).

Grep for `TODO(GNR):` in this directory to find the open work.

## What this will be (post-scaffold)

A single PyBind11 C++ extension (`_native`) exporting:

```python
amx_fp16_attn._native.flash_attn_fp16(
    Q, K, V,         # (B, H, S, D) fp16, contiguous, CPU
    sm_scale: float, # usually 1 / sqrt(D)
    is_causal: bool,
) -> Tensor          # (B, H, S, D) fp16
```

Shapes supported: `D in {64, 128}`, `S` a positive multiple of 32, `B >= 1`,
`H >= 1`.  Q/K/V must share shape, dtype (`fp16`), and contiguity.  The
contract is the FP16 mirror of the BF16 sibling's `flash_attn_bf16`.

## Hardware / OS requirements

* Intel **Granite Rapids** CPU — `family=6, model=173 (0xAD)` per the
  `_INTEL_MODEL_TO_ARCH` table in `src/repercep/backend/cpu.py`.  `amx_fp16`
  must appear in `/proc/cpuinfo` flags — `setup.py` aborts otherwise.
* Linux **>= 5.16** (for `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`).
* gcc **>= 14** for `-march=graniterapids`; gcc 13 + binutils 2.41 works via
  the explicit `-mamx-fp16` sub-feature flag that `setup.py` also passes.
* PyTorch with the matching C++ ABI.

On SPR/EMR (the current dev hosts), `setup.py` refuses to build and the
Python wrapper's `available` property is `False` — same end-state as the
BF16 kernel on a non-AMX host today.

## Build

From this directory, on a GNR host:

```bash
python setup.py build_ext --inplace
```

That produces `_native.cpython-*.so` alongside `__init__.py`.  On SPR/EMR
the script exits with a refusal message; do not bypass it.

## Python wrapper

The user-facing API lives at:

```
src/repercep/attention/amx_fp16_flash.py
```

That module is responsible for importing `_native`, doing the dtype/device
coercion, exposing the same call signature as the BF16 sibling, and
disqualifying itself (`available is False`) on hosts without `amx_fp16`.

## Notes

* `_tile_release()` will be called at the end of each parallel region (same
  pattern as the BF16 sibling, `TODO(GNR):` once the driver lands).
* The S staging buffer (16x16 fp32 = 1 KiB) and the P pack buffer
  (32x32 fp16 = 2 KiB) live on the stack per-thread.
* The exp polynomial planned for the softmax is the same FA-2 degree-5 fit
  used in the BF16 sibling -- accuracy bound is well inside the FP16 output
  precision floor.
