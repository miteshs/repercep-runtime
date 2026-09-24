# Part 5 — The kernel

This is the bottom: `kernels/triton_kernels/fp8_flash_attn.py` — the FP8
flash-attention Triton kernel for gfx942. No abstractions, no Protocols (the
loader `src/repercep/attention/fp8_triton.py` wraps it as an `AttentionOp`). Both
axes here are tight together: the ML is "attention without materializing S²," the
systems is "FP8 MFMA + tiling + autotune."

## 5.1 Why a kernel at all (the S² wall)

Part 2: self-attention over S ≈ 109k tokens, `is_causal=False` (bidirectional).
Naively that's a 109k × 109k scores matrix **per head** (~1.2 × 10¹⁰ elements) —
you cannot materialize it. **FlashAttention** (Dao 2022/2023) computes the same
result tile-by-tile with an **online softmax**, keeping only O(S) state. That's
not an optimization here; it's the only way the forward fits in memory.

## 5.2 The algorithm — FlashAttention-2, FP8 matmuls, FP32 stats

`_fp8_flash_attn_fwd_impl` (line 164): **one program = one (batch, head, Q-tile)**.
The Q-tile stays resident; K/V tiles stream past; running statistics accumulate.

```python
# state, FP32 (accuracy stays here even though the matmuls are FP8)
m_i = -inf      # running max per query row      (BLOCK_M,)
l_i = 0         # running sum of exp              (BLOCK_M,)
acc = 0         # running output accumulator      (BLOCK_M, BLOCK_D)
qk_scale = sm_scale * q_scale * k_scale           # dequant folded into the temperature (242)

for each K/V tile (BLOCK_N):                       # loop, line 247
    k = load(FP8)                                  # (BLOCK_N, BLOCK_D)
    qk = tl.dot(q_tile, k.T, out_dtype=fp32) * qk_scale   # FP8·FP8 → FP32   (259)  ← MFMA
    qk = mask(qk)                                  # causal / partial-tile
    m_new = max(m_i, rowmax(qk))                   # online softmax          (272)
    alpha = exp(m_i - m_new)                       # rescale prior state
    p     = exp(qk - m_new)                         # (BLOCK_M, BLOCK_N), in [0,1]
    l_i   = l_i*alpha + rowsum(p)
    acc   = acc*alpha                               # rescale accumulator
    v = load(FP8)
    p_fp8 = (p * 240).to(float8e4b8)                # requantize P for the 2nd matmul (288)
    acc  += tl.dot(p_fp8, v, out_dtype=fp32) * (v_scale/240)   # FP8·FP8 → FP32  (292)  ← MFMA
    m_i = m_new
acc = acc / l_i                                     # normalize               (298)
store(acc)
```

Three details worth their own sentence:

- **FP8 in, FP32 accumulate.** Both `tl.dot`s take FP8 inputs and accumulate in
  FP32 — the matmuls are cheap/low-precision, the softmax statistics stay
  accurate. This is the FlashAttention-3 FP8 recipe (Shah et al. 2024 §3).
- **Dequant folded into the temperature** (242). Instead of dequantizing the
  scores element-by-element after the dot, the per-(B,H) `q_scale·k_scale` is
  multiplied into `sm_scale` once — one float multiply per program, not per
  element. A small, characteristic kernel-engineering move.
- **P is requantized to FP8** for the second matmul with a *fixed* scale of 240
  (P ∈ [0,1], so 240 saturates the e4m3 range), and the 1/240 dequant is folded
  into `v_scale` (288-293).

The Python wrapper `fp8_flash_attention` (393) does the quantization around the
launch: `_per_bh_scale` (338) picks a dequant scale per (B,H) as
`amax / (0.95·240)`; `_quantize` (352) casts BF16 → `float8_e4m3fnuz`.

## 5.3 How `tl.dot` lowers to the ISA (the "down to the kernel" answer)

This is the literal bottom of "how does it get lower to the kernel." From the
file header (13-18):

> Triton on ROCm targets gfx942's MFMA pipeline through MLIR; the compiler emits
> `v_mfma_f32_*_fp8_fp8` for FP8 `tl.dot` on this architecture.

So the chain is: **Python `tl.dot(fp8, fp8, out=fp32)` → Triton IR → MLIR → LLVM
AMDGPU → the `v_mfma_f32_16x16x32_fp8_fp8` matrix-core instruction** on each CDNA3
compute unit. Hand-writing that in HIP means calling
`__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8` yourself plus LDS double-buffering —
"six weeks of perf engineering" vs "an afternoon" for Triton (and exactly what the
in-tree `kernels/hip/fp8_attn/` proof-of-life attempts, and gets *wrong* — its
B-operand load under-samples K). Triton is how
Repercep gets 80% of hand-tuned MFMA for an afternoon's work.

## 5.4 Autotune — why one tile shape isn't enough (F20 → F21)

The kernel's tile (`BLOCK_M`, `BLOCK_N`) and launch meta (`num_warps`,
`num_stages`) have a **shape-dependent** sweet spot. The project learned this the
hard way:

- A fixed tile (128/64) won **1.92×** at the *benchmark* shape (B=1, H=8 — an
  8-column program grid that amortizes the tile) — F19/Session 9.
- The *same* fixed tile was **0.98× (a wash)** at Cosmos's production shape (B=2,
  H=32 — a 64-column grid) — **F20**.
- Autotuning over a constrained grid (`_autotune_configs`, 110) recovered the win:
  **~1.13–1.16×** at the production shape — **F21**. That's the modest ~9 s the FP8
  path shaves to reach the 142 s MI300X headline.

`@triton.autotune` (331) searches the grid keyed on `(Sq, Skv, BLOCK_D, H,
CAUSAL)`; a **persistent JSON cache** (`~/.cache/repercep/fp8_autotune.json`, 57-94)
records the winner per `(B,H,Sq,Skv,D,causal)` so it's one search per shape per
host, then free across processes (atomic tmp-rename write, corruption-tolerant
load). `fp8_flash_attention` (393) resolves: env-disable → fallback tile;
cache-hit → launch fixed; miss → autotune then persist.

## 5.5 Honest perf (so you read the headline correctly)

This is a real, working FA-2 FP8 kernel — and a **modest** win on AMD (~1.13×
SDPA→aotriton at the Cosmos shape) and a **loss** on Hopper (cuDNN-FA3 is a far
harder bar — F27/F29, which is why Part 4's `=fa` mode exists). Note
that the headline speedup is the *cache* (Part 3), not
this kernel; at the kernel level Repercep is at parity-or-behind. Knowing that is
the difference between reading "3.81×" correctly and overclaiming it.

## Run it (CPU; the kernel itself needs a GPU)

The numerical kernel needs gfx942 + Triton-ROCm. But the autotune-cache plumbing
and grid logic are CPU-testable:

```bash
python -m pytest tests/test_attention_fp8.py -q -k "autotune or cache or grid"
# on a GPU box, exercise the real kernel end-to-end:
#   REPERCEP_FP8_ATTENTION=1 python scripts/run_cosmos.py --frames 121 --steps 36 --native-loop --cache-mode adaptive
```

**Next:** Part 6 — the clean latent becomes pixels (VAE decode), then a streamed
`Frame`.
