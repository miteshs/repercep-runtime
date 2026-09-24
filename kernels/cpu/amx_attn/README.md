# `amx_attn` — Intel AMX BF16 flash-attention for Repercep

CPU sibling of the GPU Triton flash-attention kernels (`kernels/gpu/...`).
Implements a FlashAttention-2 forward pass using Intel AMX BF16 tile
instructions (`TDPBF16PS`) plus AVX-512 BF16/FP32 for the online softmax and
output rescale.

## What this is

A single PyBind11 C++ extension (`_native`) exporting:

```python
amx_attn._native.flash_attn_bf16(
    Q, K, V,         # (B, H, S, D) bf16, contiguous, CPU
    sm_scale: float, # usually 1 / sqrt(D)
    is_causal: bool,
) -> Tensor          # (B, H, S, D) bf16
```

Shapes supported: `D in {64, 128}`, `S` a positive multiple of 32, `B >= 1`,
`H >= 1`. Q/K/V must share shape, dtype (`bf16`), and contiguity.

The kernel does:

1. Pre-pack `K` once into the AMX B-operand layout `(B, H, D/2, S, 2)`.
2. Pre-pack `V` once into the AMX B-operand layout `(B, H, S/2, D, 2)`.
3. OpenMP-parallel outer loop over `(B, H, q_tile)` with one `_tile_loadconfig`
   per thread. Inner loop tiles over `S_kv` and applies the online softmax in
   AVX-512, early-outing on causal masking.
4. Final divide by row-sum, cast to BF16, store.

## Hardware / OS requirements

* Intel **Sapphire Rapids** (or newer) CPU. `amx_bf16` must appear in
  `/proc/cpuinfo` flags — `setup.py` aborts otherwise.
* Linux **>= 5.16** (for `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`).
* gcc **>= 12** (Ubuntu 24.04 ships 13, which works out-of-the-box).
* PyTorch with the matching C++ ABI.

## Build

From this directory:

```bash
python setup.py build_ext --inplace
```

That produces `_native.cpython-*.so` alongside `__init__.py`.

## Python wrapper

The user-facing API lives at:

```
src/repercep/attention/amx_flash.py
```

That module is responsible for importing `_native`, doing the dtype/device
coercion, exposing the same call signature as the GPU Triton FA kernels, and
falling back to a reference path on hosts that lack AMX.

## Notes

* `_tile_release()` is called at the end of each parallel region.
* The S staging buffer (16x16 fp32 = 1 KiB) and the P pack buffer (32x32 bf16
  = 2 KiB) live on the stack per-thread.
* The exp polynomial is the standard FlashAttention-2 degree-5 fit on the
  range `[-ln 2 / 2, ln 2 / 2]`, accurate to about 2 ULP — good enough for a
  bf16 output.
