# kernels/ — native compute kernels (Triton / CUDA / HIP)

**Empty until we move off SDPA→aotriton.** Today the MI300X attention path
goes through PyTorch's `scaled_dot_product_attention`, which on ROCm 7.2
routes through aotriton flash kernels under the hood (see
`docs/adr/0002-rocm-attention-primitive.md`). That is fast enough for v0.1.
When we need shape-specific or fusion-specific kernels we don't get from
SDPA's autorouting, they land here.

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
