# kernels/ — native compute kernels (Triton / CUDA / HIP)

**Populated.** The default MI300X attention path still goes through PyTorch's
`scaled_dot_product_attention`, which on ROCm 7.2 routes to aotriton flash
kernels under the hood (see `docs/adr/0002-rocm-attention-primitive.md`).
What lives here are the ops that beat or extend that floor:

| Path | What | Status |
|---|---|---|
| `triton_kernels/fp8_flash_attn.py` | FP8 FA-2 for gfx942 | production, opt-in via `REPERCEP_FP8_ATTENTION=1` |
| `triton_kernels/fp8_flash_attn_hopper.py` | sm_90a sibling | slower than cuDNN-FA3 today (F29, F42) |
| `triton_kernels/fp8_flash_attn_ada.py` | sm_89 sibling | see F44 |
| `cpu/amx_*` | AMX attention (bf16 / fp16 / int8) | production on Sapphire Rapids |
| `hip/fp8_attn/` | MFMA FP8 GEMM | **toolchain proof-of-life only — not wired to any caller** |

`hip/fp8_attn/` deserves its warning label: it exists so the HIP toolchain is
proven end-to-end (kernel → pybind → `AttentionOp`), not because anything calls
it. Its operand register layout is pinned by `tests/test_mfma_fragment_layout.py`
after F48 found it loading on 16 of 64 lanes. Promoting it to a perf path needs
an LDS staging rewrite; until then the FP8 perf path is the Triton kernel.

## Why this is at the repo root, not under src/

Per the implementation plan's "Architectural Choices to Maximize AI
Leverage" (§3.4): "Separate kernel layer from architecture layer. The kernel
layer needs to violate clean architecture principles for performance reasons.
Mixing the two produces either clean-but-slow or fast-but-unmaintainable
code." This directory is treated as a separate codebase culturally:

- **No `mypy --strict`**, no `ruff` lints. Triton bodies are eval-loaded at
  runtime and don't yield helpful annotations; CUDA / HIP `.cu`/`.hip` are
  compiled by `hipcc`/`nvcc`.
- **No `Protocol` abstractions**, no `Pydantic` models, no
  `cargo clippy -D warnings`. Perf-first.
- **No imports from `src/repercep/`** — the seam between Repercep proper and a
  kernel is the `AttentionOp` Protocol (and future siblings). Kernels are
  loaded behind that seam; they don't reach back the other way.

## Future layout (when populated)

```
kernels/
  triton/        # Triton-as-Python kernels (DiT attention variants, fused
                 # action conditioning, etc.). Loaded by src/repercep/attention/.
  cuda/          # CUDA C++ for Hopper/Blackwell features Triton doesn't expose
                 # (TMA, warp specialization, cluster launch). NVIDIA backend
                 # only; deferred to the NVIDIA fast-follow.
  hip/           # HIP C++ for AMD MI300X / CDNA3 — MFMA tiles, FP8, etc.
  build.cmake    # Single CMake entry point for the C++/HIP/CUDA targets.
                 # Cargo + uv stay the user-facing build drivers; CMake is
                 # called from them for kernel builds only.
```

CMake is intentionally only here, not at the repo root. The handoff doc's
build-tooling line — "Cargo + CMake + uv. Resist Bazel" — slots CMake into
this directory exclusively. Cargo handles the Rust workspace; uv handles
Python; CMake handles native kernels.

## Adding a kernel

The path is roughly:

1. Implement the kernel in `kernels/triton/<name>.py` (or `kernels/hip/<name>.hip`).
2. Write a thin loader in `src/repercep/attention/<name>.py` (or the relevant
   subsystem) that implements `AttentionOp` and calls the kernel.
3. Register it in `src/repercep/attention/registry.py`.
4. Benchmark vs the SDPA floor with `scripts/profile_cosmos.py --compare`.
   The win has to be measurable on the *full reference config* (121 f /
   36 steps), not a tiny shape, before promoting it to default.

See `docs/OPTIMIZATION.md` for the measurement ledger.
