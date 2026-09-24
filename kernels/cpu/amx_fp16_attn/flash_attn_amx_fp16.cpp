// =============================================================================
//  Repercep CPU sibling kernels: flash-attention on Intel AMX FP16 (Granite Rapids)
// =============================================================================
//
//  This is the FP16 sibling of `kernels/cpu/amx_attn/flash_attn_amx.cpp`.
//  Granite Rapids (GNR) introduces `AMX_FP16` -- the first Xeon silicon that
//  carries an FP16 tile-matmul instruction (`TDPFP16PS` aka the
//  `_tile_dpfp16ps` intrinsic).  On SPR/EMR the BF16 sibling is the right
//  AMX path; on GNR this kernel becomes the FP16 fast path and the BF16
//  sibling continues to handle BF16 inputs.
//
//  -------------------------------------------------------------------------
//  Scaffold status (Item C of the CPU port)
//  -------------------------------------------------------------------------
//
//  This file is a **scaffold**, not a finished kernel.  It compiles cleanly
//  on a real GNR host (where `-march=graniterapids` / `-mamx-fp16` are
//  accepted by gcc 14+) and the C++ shape/dtype contract matches the BF16
//  sibling exactly.  The forward currently delegates to PyTorch's
//  SDPA fallback so that downstream callers can wire the wrapper end-to-end
//  before the optimised tile loops are filled in.
//
//  The algorithmic skeleton -- online softmax, OMP outer loop, per-thread
//  tile config -- is laid out below as `// TODO(GNR):` markers so that the
//  follow-up work has an exact map of what needs to land:
//
//      1. Pre-pack K into AMX B-layout for FP16 (analog of pack_K_for_amx_B).
//      2. Pre-pack V into FP16-pair layout (analog of pack_V_pairs).
//      3. FlashAttention-2 online-softmax inner loop driven by
//         `_tile_dpfp16ps` for both the Q@K^T and P@V matmuls.
//      4. AVX-512 FP16 (`avx512_fp16`) fast paths for the softmax / cast
//         steps -- GNR carries the full AVX-512_FP16 ISA so the cast back
//         to FP16 can use the native intrinsics rather than the bit-twiddle
//         used in the BF16 sibling.
//
//  All four items follow once GNR hardware (or the GNR-class SDE model) is
//  reachable for end-to-end numerical correctness checks.  Until then the
//  SDPA fallback keeps the call-graph honest.
//
//  -------------------------------------------------------------------------
//  Algorithm at a glance (mirrors the BF16 sibling)
//  -------------------------------------------------------------------------
//
//    For each (batch, head) and Q-tile of M_Q=32 rows:
//        m_i  = -inf                                # running rowmax
//        l_i  =  0                                  # running rowsum
//        O_i  =  0                                  # running output acc
//        for each K-tile of N_KV=32 columns (early-out on causal):
//            S = Q_tile @ K_tile^T                  # AMX FP16 -> FP32
//            S *= sm_scale
//            (optional causal mask)
//            m_new = max(m_i, rowmax(S))
//            alpha = exp(m_i - m_new)
//            P     = exp(S   - m_new)               # same row broadcast
//            l_i   = alpha * l_i  + rowsum(P)
//            O_i   = alpha * O_i  + P @ V_tile      # AMX FP16 -> FP32
//            m_i   = m_new
//        O_i /= l_i
//        cast O_i to FP16, store.
//
//  -------------------------------------------------------------------------
//  AMX tile layout for FP16 (palette = 1)
//  -------------------------------------------------------------------------
//
//    Each AMX tile may carry up to 16 rows and 64 bytes/row (= 1 KiB).
//    For TDPFP16PS, FP16 inputs are packed as 16 rows x (32 fp16) = 64 B/row,
//    and the FP32 accumulator is 16 rows x (16 fp32) = 64 B/row -- the byte
//    geometry is bit-identical to the BF16 case, which is why we can reuse
//    the BF16 sibling's `TileConfig` verbatim.  The instruction differs:
//
//        BF16 sibling: _tile_dpbf16ps(TMM_S, TMM_Q, TMM_K)
//        FP16 (this):  _tile_dpfp16ps(TMM_S, TMM_Q, TMM_K)
//
// =============================================================================

#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <c10/util/Half.h>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// XFEATURE / ARCH_PRCTL plumbing.  Linux requires user-space to opt-in to the
// AMX TILE state via arch_prctl before touching any TMM register.  Without
// this, the first tile instruction faults with SIGILL.  Same dance as the
// BF16 sibling -- AMX_FP16 lives under the same XSAVE feature bit.
// ---------------------------------------------------------------------------
#include <sys/syscall.h>
#include <unistd.h>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

namespace {

// One-shot opt-in for AMX tile data.  Safe to call concurrently; the kernel
// returns 0 on second-and-subsequent calls.
bool enable_amx_once() {
    static bool ok = []() {
        long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                          XFEATURE_XTILEDATA);
        return rc == 0;
    }();
    return ok;
}

// ---------------------------------------------------------------------------
// AMX tile configuration (palette 1) -- identical geometry to BF16 because
// FP16 lanes are also 2 bytes wide.
// ---------------------------------------------------------------------------
struct alignas(64) TileConfig {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[16];   // bytes per row
    uint8_t  rows[16];    // number of rows
};

// Tile assignments.  See BF16 sibling for the macros-vs-constexpr rationale.
#define TMM_Q 0   // 16 x 32 fp16  (Q sub-tile)
#define TMM_K 1   // 16 x 32 fp16  (K^T sub-tile)
#define TMM_S 2   // 16 x 16 fp32  (S = Q @ K^T accumulator)
#define TMM_P 3   // 16 x 32 fp16  (P sub-tile fed to second matmul)
#define TMM_V 4   // 16 x 32 fp16  (V sub-tile -- fp16 pair packed)
#define TMM_O 5   // 16 x 16 fp32  (O accumulator, one head_dim col group)

// Set up the tile config for a single thread.  Must be called once per thread
// before any AMX instruction.  Layout is byte-identical to the BF16 sibling.
[[maybe_unused]] void configure_tiles_for_thread() {
    TileConfig cfg = {};
    cfg.palette_id = 1;
    cfg.start_row  = 0;

    // fp16 inputs: 16 rows x 32 fp16 = 16 x 64 B
    cfg.rows[TMM_Q] = 16;  cfg.colsb[TMM_Q] = 64;
    cfg.rows[TMM_K] = 16;  cfg.colsb[TMM_K] = 64;
    cfg.rows[TMM_P] = 16;  cfg.colsb[TMM_P] = 64;
    cfg.rows[TMM_V] = 16;  cfg.colsb[TMM_V] = 64;

    // fp32 accumulators: 16 rows x 16 fp32 = 16 x 64 B
    cfg.rows[TMM_S] = 16;  cfg.colsb[TMM_S] = 64;
    cfg.rows[TMM_O] = 16;  cfg.colsb[TMM_O] = 64;

    _tile_loadconfig(&cfg);
}

// ---------------------------------------------------------------------------
// SDPA reference fallback.
//
// While the AMX_FP16 inner loop is still being filled in (see TODO(GNR)
// markers below) the entry point routes through `at::scaled_dot_product_attention`
// so that callers see a numerically-correct result and the wrapper's import
// path / dispatch contract stays exercised.  On a real GNR host this will be
// replaced by the AMX path below.
// ---------------------------------------------------------------------------
at::Tensor sdpa_fallback(const at::Tensor& Q,
                         const at::Tensor& K,
                         const at::Tensor& V,
                         double sm_scale,
                         bool is_causal) {
    // PyTorch's SDPA accepts an explicit scale, so we don't have to bake
    // 1/sqrt(D) into Q ourselves.  attn_mask is left as nullopt; is_causal
    // takes care of the triangular case.
    return at::scaled_dot_product_attention(
        Q, K, V,
        /*attn_mask=*/c10::nullopt,
        /*dropout_p=*/0.0,
        /*is_causal=*/is_causal,
        /*scale=*/sm_scale);
}

// =========================================================================
// TODO(GNR): Pre-pack K into the AMX "B-operand" layout for FP16.
//
// The BF16 sibling's `pack_K_for_amx_B` lays out K as (B, H, D/2, S, 2)
// because TDPBF16PS reads its B operand as 16 rows x (32 BF16 = 16 pairs).
// TDPFP16PS uses the identical 16x32 byte geometry, so the FP16 packing
// shape is the same -- only the element type changes.  Implementation is
// a near-verbatim copy of the BF16 routine swapping `c10::BFloat16` for
// `c10::Half`.
// =========================================================================
// TODO(GNR): at::Tensor pack_K_for_amx_B_fp16(const at::Tensor& K);

// =========================================================================
// TODO(GNR): Pre-pack V into AMX FP16-pair layout.
//
// Same shape transform as the BF16 sibling's `pack_V_pairs`, FP16 element
// type instead of BF16.  Output shape: (B, H, S/2, D, 2).
// =========================================================================
// TODO(GNR): at::Tensor pack_V_pairs_fp16(const at::Tensor& V);

// =========================================================================
// TODO(GNR): Core per-Q-tile worker for FP16.
//
// The algorithmic shell is identical to the BF16 sibling's `process_q_tile`:
//
//   * Per-thread `configure_tiles_for_thread()` in the outer OMP region.
//   * Online softmax state: row_max[M_Q], row_sum[M_Q], O_acc[M_Q * D].
//   * Inner loop over kv_tiles of N_KV=32:
//       - Build S = Q @ K^T via four _tile_dpfp16ps calls (covers the
//         (32 Q rows x 32 KV cols) S block in two row-halves and two
//         col-halves; the per-call K-dim chunk is 32 FP16 == 64 B per row).
//       - AVX-512 softmax with causal mask + degree-5 exp polynomial.
//         GNR carries AVX-512_FP16; the cast back to FP16 can use
//         `_mm512_cvtps_ph` directly rather than the BF16 bit-shift trick.
//       - Pack P into the FP16 A-operand layout (row-major, 16 x 32).
//       - O += P @ V via _tile_dpfp16ps over D/16 column tiles.
//   * Final divide-by-row-sum, cast FP32 -> FP16, store.
//
// The signature below is the contract the eventual implementation has to
// hit -- it matches the BF16 sibling 1:1 with the element type swapped.
// =========================================================================
// TODO(GNR): void process_q_tile_fp16(
//                const c10::Half* __restrict Q_tile_ptr,
//                const c10::Half* __restrict K_pk_ptr,
//                const c10::Half* __restrict V_p_ptr,
//                c10::Half* __restrict       O_tile_ptr,
//                int64_t S_kv, int64_t D, int64_t q_start,
//                float sm_scale, bool is_causal);

// =========================================================================
// TODO(GNR): Replace `sdpa_fallback` below with the real AMX_FP16 driver:
//
//   * One-shot `enable_amx_once()` (already wired).
//   * Pre-pack K and V once per call.
//   * #pragma omp parallel { configure_tiles_for_thread();
//                            #pragma omp for collapse(3) ...
//                                process_q_tile_fp16(...);
//                            _tile_release(); }
// =========================================================================

}  // namespace


// =========================================================================
// PyBind entry point.
//
// Shape / dtype contract is the FP16 mirror of `flash_attn_bf16`.  Until the
// AMX_FP16 inner kernel lands, the implementation delegates to
// `at::scaled_dot_product_attention` so that callers see correct numerics on
// the (currently inaccessible) GNR target.
// =========================================================================
at::Tensor flash_attn_fp16(const at::Tensor& Q,
                           const at::Tensor& K,
                           const at::Tensor& V,
                           double sm_scale,
                           bool is_causal) {
    // ---- Shape/dtype/contiguity asserts.
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "Q/K/V must be 4D (B, H, S, D)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q/K/V shapes must match for this kernel");
    TORCH_CHECK(Q.scalar_type() == at::kHalf, "Q must be fp16");
    TORCH_CHECK(K.scalar_type() == at::kHalf, "K must be fp16");
    TORCH_CHECK(V.scalar_type() == at::kHalf, "V must be fp16");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q/K/V must be contiguous");
    TORCH_CHECK(Q.device().is_cpu(), "this is the CPU/AMX kernel");

    const int64_t D = Q.size(3);
    const int64_t S = Q.size(2);
    TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128, got ", D);
    TORCH_CHECK(S >= 32 && S % 32 == 0,
                "sequence length must be a positive multiple of 32, got ", S);

    // ---- One-shot AMX permission opt-in.  Required even by the scaffold
    // call so that the wired-up path on a real GNR host fails fast at the
    // permission step rather than at the first `_tile_dpfp16ps`.
    TORCH_CHECK(enable_amx_once(),
                "Linux refused arch_prctl(ARCH_REQ_XCOMP_PERM, "
                "XFEATURE_XTILEDATA). Kernel >= 5.16 with AMX support is "
                "required.");

    // TODO(GNR): replace with the real AMX_FP16 driver once tile loops land.
    return sdpa_fallback(Q, K, V, sm_scale, is_causal);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fp16", &flash_attn_fp16,
          "Flash attention on AMX FP16 (Granite Rapids; scaffold)");
}
