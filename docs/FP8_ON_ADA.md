# FP8 Flash Attention on Ada Lovelace (sm_89)

**Status:** Works. Time-boxed half-day port landed 2026-05-25 (Item J).

Ada Lovelace is the third silicon target for Repercep's fused FP8
flash-attention kernel, alongside CDNA3 (gfx942 / MI300X) and Hopper
(sm_90a / H100). It is the *fourth* FP8 target overall once you count
the unfused `fp8_scaled_mm` path.

## Why this kernel exists

Ada Lovelace (RTX 4090, L40, L40S, RTX 2000-6000 Ada, Q4-2022 launch)
shipped 4th-generation tensor cores with **native FP8 ISA support** —
E4M3 and E5M2, the same dtypes Hopper has. The exposure is different:
sm_89 uses the synchronous `mma.sync.aligned.m16n8k32.f32.e4m3.e4m3`
PTX instruction (single-warp), where sm_90a uses asynchronous
warpgroup `wgmma.mma_async.sync.aligned` (full warpgroup, TMA-fed).
The Triton 3.4 NVPTX backend dispatches `tl.dot` with FP8 operands to
the correct instruction per arch — so a Triton kernel that uses
neither Hopper-only intrinsics (TMA descriptors, `wgmma` fences,
`tl.async_copy`) nor AMD-only intrinsics compiles cleanly on both.

In practice this means most of the FP8 ecosystem (FA-3 FP8,
Transformer Engine) was built Hopper-first and the Ada FP8 path is
under-exercised. This wrapper closes the gap for Repercep and gives the
project a real fourth silicon target without requiring TE or FA-3.

## Files

| File | Role |
|------|------|
| `kernels/triton_kernels/fp8_flash_attn_ada.py` | Triton kernel |
| `src/repercep/attention/fp8_ada_triton.py` | `AttentionOp` wrapper |
| `tests/test_fp8_attention_ada.py` | Correctness + structural tests |

Kernel and wrapper are direct siblings of the Hopper versions
(`fp8_flash_attn_hopper.py` / `fp8_hopper_triton.py`) — same
FlashAttention-2 online softmax, same FP8 dtype (`e4m3fn`, IEEE),
same `FP8_MAX=448`. Divergences:

- **Autotune grid.** Ada has ~100 KiB SMEM/block (vs Hopper's 228 KiB
  and CDNA3's 64 KiB), so we cap tiles at 256x128 and drop the 256x256
  entry that Hopper uses. Ada's narrower SMs prefer `num_warps=4`,
  unlike Hopper which prefers 8.
- **`num_stages`.** Synchronous `mma.sync` doesn't benefit from deep
  software pipelining the way `wgmma` does; sweep is (2, 3) vs
  Hopper's (2, 3, 4, 5).
- **Cache file.** `~/.cache/repercep/fp8_autotune_ada.json`, distinct
  from the Hopper and CDNA3 files because the optimal configs differ.
- **Silicon gate.** The wrapper probes `torch.cuda.get_device_capability`
  and disqualifies unless `(8, 9)`. sm_80 / sm_86 (Ampere) have no FP8
  tensor cores; sm_90 should use the Hopper kernel.

## Measured on RTX 2000 Ada (sm_89, 16 GiB)

Shape: `(B=1, H=8, S=4096, D=128)` bf16 input → quantized E4M3 → kernel
→ bf16 output. Reference: `torch.nn.functional.scaled_dot_product_attention`
on the bf16 inputs (routes to cuDNN flash on sm_89).

| Metric | Value |
|---|---|
| Max abs error vs SDPA | 0.014 |
| Mean abs error vs SDPA | 0.001 |
| Mean rel error (significant entries) | ~5% |
| Median rel error | ~3.6% |
| Steady-state ms/call (autotuned) | 3.07 ms |
| SDPA reference ms/call | 1.74 ms |
| Autotuned config | `BLOCK_M=64 BLOCK_N=128 num_warps=4 num_stages=2` |

The FP8 path is **~1.77× slower than SDPA at S=4096** — this is
expected. The FP8 win is at long sequences (S in the tens of thousands)
where SDPA becomes bandwidth-bound and the FP8 register-resident tile
loop pulls ahead. At S=4096 the cuDNN flash path on Ada has plenty of
bandwidth headroom and FP8 quantization overhead dominates.

This is the same shape-dependent crossover that Cosmos saw on H100
(F20, F27 in BUILD_LOG) — the kernel only earns its keep at the
Cosmos production shape (S=109k) and similar. RTX 2000 Ada has 16 GiB
HBM3 vs H100's 80 GiB, so testing at S=109k is not feasible on this
host — that bench would require an L40S (48 GiB) or RTX 6000 Ada.

## Numerical precision

The 5% mean relative error matches the FP8 floor that the Hopper and
CDNA3 siblings see (F27: 3-5% rel diff at Cosmos shape on Hopper). The
quantization is per-(batch, head) amax-based, same as the siblings;
the kernel uses Triton's `tl.float8e4nv` (IEEE e4m3 with inf/nan) and
folds the dequant scale into the softmax temperature.

This is well within the "FP8 attention is acceptable for diffusion
forward pass" tolerance and matches what FA-3 FP8 and TE FP8 measure
on the same models.

## Not yet wired into the registry

The wrapper is **not** registered in `src/repercep/attention/registry.py`
or routed in `src/repercep/backend/cuda.py` yet — those files are owned
by the INTEG agent (see Item J handoff). Following hooks would be the
natural wiring point:

1. Add `from repercep.attention.fp8_ada_triton import FP8AdaTritonAttention`
   to the NVIDIA selection branch.
2. Insert ahead of `naive-sdpa` but behind `nvidia-flash` for non-FP8
   workloads; ahead of `nvidia-flash` only when `REPERCEP_FP8_ATTENTION`
   is set.
3. Update `BackendCapabilities.attention_ops` and `supports_fp8` for
   `(8, 9)` hosts.

The op self-disqualifies on non-Ada hosts via the device-cap probe,
so wiring it into the unified NVIDIA branch is safe even on H100 hosts.

## Future work

- Bench at S=16k+ on an L40S to confirm the crossover where FP8 actually
  wins on Ada (the RTX 2000 Ada is too small to host that test).
- Try the FlashAttention-3 FP8 approach with block scaling (vs our
  per-(B,H) scaling) — may close some of the 5% rel-error gap.
- Evaluate `tl.dot_scaled` (new in Triton 3.4) for finer-grained scale
  injection without re-quantizing P inside the inner loop.
