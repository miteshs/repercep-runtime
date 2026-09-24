// =============================================================================
//  Repercep CPU sibling kernels: flash-attention on Intel AMX BF16
// =============================================================================
//
//  This file implements a FlashAttention-2 style scaled-dot-product-attention
//  forward pass that targets Intel Sapphire Rapids+ via the AMX BF16 tile
//  instructions (TDPBF16PS), AVX-512 BF16 (VCVTNE2PS2BF16), and AVX-512F.
//
//  It is the CPU counterpart of the GPU Triton kernels that live under
//  `kernels/gpu/...`.  The shapes, dtype contract, and call signature mirror
//  what the Triton kernel exposes so that the Python wrapper in
//  `src/repercep/attention/amx_flash.py` can swap them transparently.
//
//  -------------------------------------------------------------------------
//  Algorithm at a glance
//  -------------------------------------------------------------------------
//
//    For each (batch, head) and Q-tile of M_Q=32 rows:
//        m_i  = -inf                                # running rowmax
//        l_i  =  0                                  # running rowsum
//        O_i  =  0                                  # running output acc
//        for each K-tile of N_KV=32 columns (early-out on causal):
//            S = Q_tile @ K_tile^T                  # AMX BF16 -> FP32
//            S *= sm_scale
//            (optional causal mask)
//            m_new = max(m_i, rowmax(S))
//            alpha = exp(m_i - m_new)
//            P     = exp(S   - m_new)               # same row broadcast
//            l_i   = alpha * l_i  + rowsum(P)
//            O_i   = alpha * O_i  + P @ V_tile      # AMX BF16 -> FP32
//            m_i   = m_new
//        O_i /= l_i
//        cast O_i to BF16, store.
//
//  -------------------------------------------------------------------------
//  AMX tile layout (palette = 1)
//  -------------------------------------------------------------------------
//
//    Each AMX tile may carry up to 16 rows and 64 bytes/row (= 1 KiB).
//    For TDPBF16PS, BF16 inputs are packed as 16 rows x (32 bf16) = 64 B/row,
//    and the FP32 accumulator is 16 rows x (16 fp32) = 64 B/row.
//
//    We use four tiles per inner step:
//        tmm0 := Q sub-tile   (16 x 32 bf16, row stride = D * 2 bytes)
//        tmm1 := K^T sub-tile (16 x 32 bf16, row stride = S_kv * 2 bytes)
//        tmm2 := S accum      (16 x 16 fp32, row stride = 16 * 4 = 64 B)
//        tmm3 := scratch / V / O sub-tile depending on phase
//
//    Because a single AMX tile only covers 16 query rows but the spec asks
//    for M_Q=32 queries per Q-tile, we stripe each Q-tile into TWO 16-row
//    halves and run the inner sequence twice per K-tile.
//
//  -------------------------------------------------------------------------
//  Optimisation notes (durable, in-source so they survive grep)
//  -------------------------------------------------------------------------
//
//    * K is re-packed once per (B, H) into K_T shape (B, H, D, S_kv) so the
//      AMX tile loads are contiguous in the inner loop.  Cost: O(B*H*S*D)
//      moves, amortised over O(B*H*S^2/T_kv) AMX matmuls -- well worth it.
//    * V is re-packed once per (B, H) into a BF16-pair layout
//      (B, H, S_kv/2, D, 2) so the second matmul `_tile_loadd` reads each
//      pair of consecutive rows as the (k, k+1) BF16 columns AMX expects.
//    * The S staging buffer (16x16 fp32 = 1 KiB) and the P staging buffer
//      (16x32 bf16 = 1 KiB) are stack-allocated, 64-byte aligned, per thread.
//    * On is_causal, we skip K-tiles whose first K index is strictly past
//      the last Q index in the current Q-tile.  Inside the boundary tile
//      we apply a fine-grained -inf mask in AVX-512.
//    * We use `_mm512_cvtne2ps_pbh` to cast pairs of fp32 zmms to a single
//      bf16 zmm.  This is the AVX-512 BF16 cast intrinsic; on gcc 13 it
//      requires `-mavx512bf16` and `<immintrin.h>` only.
//
// =============================================================================

#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>

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
// this, the first tile instruction faults with SIGILL.
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
// AMX tile configuration (palette 1).
// ---------------------------------------------------------------------------
struct alignas(64) TileConfig {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[16];   // bytes per row
    uint8_t  rows[16];    // number of rows
};

// Tile assignments used throughout the kernel.
//
// IMPORTANT: these MUST be preprocessor macros, not `constexpr int`, because
// gcc's `_tile_loadd` / `_tile_stored` / `_tile_dpbf16ps` / `_tile_zero`
// intrinsics are implemented as macros that stringize the tile-number
// argument with `#dst` to bake `%%tmm<N>` into an inline-asm template.
// `constexpr int` would arrive at the stringize step as the literal token
// `TMM_O` and produce `%%tmmTMM_O`, which the GNU assembler rejects.
// Preprocessor macros are fully expanded before the inner intrinsic macro
// reaches its `#dst` step, so e.g. `_tile_loadd(TMM_O, ...)` becomes
// `__asm__ volatile ("...%%tmm5...")` as required.
#define TMM_Q 0   // 16 x 32 bf16  (Q sub-tile)
#define TMM_K 1   // 16 x 32 bf16  (K^T sub-tile -- 32 K rows x 16 outputs)
#define TMM_S 2   // 16 x 16 fp32  (S = Q @ K^T accumulator)
#define TMM_P 3   // 16 x 32 bf16  (P sub-tile fed to second matmul)
#define TMM_V 4   // 16 x 32 bf16  (V sub-tile -- bf16 pair packed)
#define TMM_O 5   // 16 x 16 fp32  (O accumulator, one head_dim col group)

// Set up the tile config for a single thread.  Must be called once per thread
// before any AMX instruction.  We configure all 6 tiles up front; unused
// tiles cost nothing.
void configure_tiles_for_thread() {
    TileConfig cfg = {};
    cfg.palette_id = 1;
    cfg.start_row  = 0;

    // bf16 inputs: 16 rows x 32 bf16 = 16 x 64 B
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
// Helpers: BF16 <-> FP32 casts in AVX-512.  We rely on -mavx512bf16.
// ---------------------------------------------------------------------------

// Convert 16 BF16 lanes (lower 256 bits of a 512-bit reg) to 16 FP32 lanes.
inline __m512 cvt_bf16x16_to_fp32(__m256i bf16x16) {
    // BF16 -> FP32 == shift left by 16 bits into the high half of each fp32.
    __m512i as_i32 = _mm512_cvtepu16_epi32(bf16x16);
    as_i32 = _mm512_slli_epi32(as_i32, 16);
    return _mm512_castsi512_ps(as_i32);
}

// Convert 32 FP32 lanes (two zmms) -> 32 BF16 lanes as a raw __m512i with RNE.
// The result is bit-identical to what `_mm512_cvtne2ps_pbh` returns, just
// re-typed so callers can use ordinary integer-vector stores; some gcc
// versions wrap __m512bh in a struct that can't be reinterpret_cast'd cleanly.
inline __m512i cvt_fp32x32_to_bf16i(__m512 a_lo, __m512 b_hi) {
    __m512bh bf = _mm512_cvtne2ps_pbh(b_hi, a_lo);
    __m512i out;
    std::memcpy(&out, &bf, sizeof(out));
    return out;
}

// Convert 16 FP32 -> 16 BF16 lanes as raw __m256i bytes.
inline __m256i cvt_fp32x16_to_bf16i(__m512 a) {
    __m256bh bf = _mm512_cvtneps_pbh(a);
    __m256i out;
    std::memcpy(&out, &bf, sizeof(out));
    return out;
}

// Horizontal max across 16 fp32 lanes.
inline float reduce_max_ps(__m512 v) {
    return _mm512_reduce_max_ps(v);
}
inline float reduce_add_ps(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// Vectorised exp() approximation good to ~2 ULP, suitable for softmax.
// We use the AVX-512 sequence: x in [-inf, 0]; compute 2^(x * log2(e)) via
// a degree-5 polynomial on the fractional part.  This is the standard
// FlashAttention-2 trick.
//
// The causal mask uses -inf in masked lanes; the polynomial path would
// produce NaN on -inf (round(-inf*c) -> indeterminate, sub(-inf, NaN) -> NaN),
// poisoning the row sum.  We therefore clamp x to a safe minimum below which
// expf already underflows to 0.  -87.336544 is just shy of expf's lower
// normal-range bound; everything below produces a true zero result.
inline __m512 exp_ps(__m512 x) {
    const __m512 LOG2E   = _mm512_set1_ps(1.44269504088896341f);
    const __m512 C0      = _mm512_set1_ps(1.0f);
    const __m512 C1      = _mm512_set1_ps(0.69314718056f);
    const __m512 C2      = _mm512_set1_ps(0.24022650695f);
    const __m512 C3      = _mm512_set1_ps(0.05550410866f);
    const __m512 C4      = _mm512_set1_ps(0.00961812910f);
    const __m512 C5      = _mm512_set1_ps(0.00133335581f);
    const __m512 X_MIN   = _mm512_set1_ps(-87.336544f);

    x = _mm512_max_ps(x, X_MIN);

    // y = x * log2(e); split into integer (n) and fractional (f) parts.
    __m512 y  = _mm512_mul_ps(x, LOG2E);
    __m512 n  = _mm512_roundscale_ps(y, _MM_FROUND_TO_NEAREST_INT |
                                        _MM_FROUND_NO_EXC);
    __m512 f  = _mm512_sub_ps(y, n);
    // Convert f back to natural-log fractional: g = f * ln(2)
    __m512 g  = _mm512_mul_ps(f, C1);

    // poly(g) = 1 + g + g^2/2 + g^3/6 + g^4/24 + g^5/120 (rearranged)
    __m512 p  = C5;
    p = _mm512_fmadd_ps(p, g, C4);
    p = _mm512_fmadd_ps(p, g, C3);
    p = _mm512_fmadd_ps(p, g, C2);
    p = _mm512_fmadd_ps(p, g, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, g, C0);
    p = _mm512_fmadd_ps(p, g, C0);

    // Scale by 2^n via bit manipulation: ldexp(p, n)
    __m512i ni = _mm512_cvtps_epi32(n);
    ni = _mm512_slli_epi32(_mm512_add_epi32(ni, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(ni));
}

// =========================================================================
// Pre-pack K into the AMX "B-operand" layout used by the Q @ K^T matmul.
//
// TDPBF16PS reads its B operand as 16 rows, each row holding 32 BF16 lanes,
// arranged so that the K-axis (here = D, the feature dim) advances 2 entries
// per row-step and 0 per column-step.  Concretely, for the logical matrix
// K^T of shape (D, S_kv), we want
//
//     K_T_pack[d/2, s, 0] = K^T[2*(d/2)  , s] = K[s, 2*(d/2)  ]
//     K_T_pack[d/2, s, 1] = K^T[2*(d/2)+1, s] = K[s, 2*(d/2)+1]
//
// Stored as a tensor of shape (B, H, D/2, S, 2) which we treat as
// (B, H, D/2, S*2) for row-stride purposes -- each AMX tile sees
// 16 rows x (16 S-cols * 2 bf16) = 1 KiB.
//
// Pre-packing once amortises the cost across all Q-tiles in the same head.
// =========================================================================
at::Tensor pack_K_for_amx_B(const at::Tensor& K) {
    // K: (B, H, S, D) bf16 contiguous.
    const int64_t B = K.size(0);
    const int64_t H = K.size(1);
    const int64_t S = K.size(2);
    const int64_t D = K.size(3);
    TORCH_CHECK(D % 2 == 0, "head_dim must be even for AMX K packing");

    auto K_pack = at::empty({B, H, D / 2, S, 2}, K.options());

    const auto* K_ptr   = reinterpret_cast<const c10::BFloat16*>(K.data_ptr());
    auto*       Kp_ptr  = reinterpret_cast<c10::BFloat16*>(K_pack.data_ptr());

    at::parallel_for(0, B * H, 1, [&](int64_t begin, int64_t end) {
        for (int64_t bh = begin; bh < end; ++bh) {
            const c10::BFloat16* src = K_ptr  + bh * S * D;
            c10::BFloat16*       dst = Kp_ptr + bh * (D / 2) * S * 2;
            for (int64_t s = 0; s < S; ++s) {
                for (int64_t d = 0; d < D; d += 2) {
                    c10::BFloat16* slot = dst + (d / 2) * S * 2 + s * 2;
                    slot[0] = src[s * D + d    ];
                    slot[1] = src[s * D + d + 1];
                }
            }
        }
    });
    return K_pack;
}

// =========================================================================
// Pre-pack V into AMX BF16-pair layout.
//
// TDPBF16PS expects the "B" matrix to be laid out so that each 4-byte cell
// holds a pair of (bf16, bf16) that sit on consecutive rows of the logical
// matrix B.  For our P @ V product, the "B" matrix is V of shape
// (S_kv, D); the packed layout we want is (S_kv/2, D, 2) with the inner-
// most 2 carrying (V[2k, d], V[2k+1, d]).
//
// V is (B, H, S, D) -> V_p is (B, H, S/2, D, 2) which we view as
// (B, H, S/2, D*2) for the AMX tile-load stride bookkeeping.
// =========================================================================
at::Tensor pack_V_pairs(const at::Tensor& V) {
    const int64_t B = V.size(0);
    const int64_t H = V.size(1);
    const int64_t S = V.size(2);
    const int64_t D = V.size(3);
    TORCH_CHECK(S % 2 == 0, "S_kv must be even for AMX V packing (got ", S, ")");

    auto V_p = at::empty({B, H, S / 2, D, 2}, V.options());

    const auto* V_ptr  = reinterpret_cast<const c10::BFloat16*>(V.data_ptr());
    auto*       Vp_ptr = reinterpret_cast<c10::BFloat16*>(V_p.data_ptr());

    at::parallel_for(0, B * H, 1, [&](int64_t begin, int64_t end) {
        for (int64_t bh = begin; bh < end; ++bh) {
            const c10::BFloat16* src = V_ptr  + bh * S * D;
            c10::BFloat16*       dst = Vp_ptr + bh * (S / 2) * D * 2;
            for (int64_t k = 0; k < S / 2; ++k) {
                const c10::BFloat16* row0 = src + (2 * k    ) * D;
                const c10::BFloat16* row1 = src + (2 * k + 1) * D;
                c10::BFloat16*       out  = dst + k * D * 2;
                for (int64_t d = 0; d < D; ++d) {
                    out[d * 2 + 0] = row0[d];
                    out[d * 2 + 1] = row1[d];
                }
            }
        }
    });
    return V_p;
}

// =========================================================================
// Core per-Q-tile worker.
//
// Each call processes ONE 32-row Q tile against the entire K dimension.
// All buffers are BF16 except the staging S/O scratches which are FP32.
//
// Args:
//   Q_tile_ptr   : (32, D)         bf16, row stride = D * 2 bytes
//   K_pk_ptr     : (D/2, S_kv, 2)  bf16, AMX B-layout (D-axis pair-packed),
//                                  row stride = S_kv * 2 * 2 bytes
//   V_p_ptr      : (S_kv/2, D, 2)  bf16, row stride = D * 2 * 2 bytes
//                                  but AMX wants row stride = D*2*2 == D*4 B
//                                  (one row of the packed layout = 16 BF16
//                                  pairs spanning 16 feature dims for two
//                                  K-rows, i.e. 64 B for D=16-strip).
//   O_tile_ptr   : (32, D)         bf16 output, row stride = D * 2 bytes
//   S_kv         : number of K tokens
//   D            : head_dim, must be 64 or 128
//   q_start      : starting Q index in the global sequence (for causal mask)
//   sm_scale     : scalar pre-softmax multiplier
//   is_causal    : enable causal masking
// =========================================================================
void process_q_tile(const c10::BFloat16* __restrict Q_tile_ptr,
                    const c10::BFloat16* __restrict K_pk_ptr,
                    const c10::BFloat16* __restrict V_p_ptr,
                    c10::BFloat16* __restrict       O_tile_ptr,
                    int64_t S_kv,
                    int64_t D,
                    int64_t q_start,
                    float sm_scale,
                    bool is_causal) {
    constexpr int M_Q  = 32;   // queries per tile (= 2 AMX tile halves)
    constexpr int N_KV = 32;   // keys per inner step  (= 2 AMX tile halves)
    constexpr int M_H  = 16;   // AMX hard-coded tile row count

    // ---- Per-row online softmax state, one fp32 entry per Q row.
    alignas(64) float row_max[M_Q];
    alignas(64) float row_sum[M_Q];
    for (int i = 0; i < M_Q; ++i) {
        row_max[i] = -std::numeric_limits<float>::infinity();
        row_sum[i] = 0.0f;
    }

    // ---- O accumulator in FP32, full (32, D).  Max footprint at D=128 is
    // 32*128*4 = 16 KiB which comfortably stays in L1d.
    alignas(64) float O_acc[M_Q * 128];
    std::memset(O_acc, 0, sizeof(float) * M_Q * D);

    // ---- Stack scratches for AMX I/O.
    // S scratch: a 16x16 fp32 tile per AMX call.  We do FOUR such tiles to
    // cover the (32 Q rows x 32 KV cols) S block (2 row halves x 2 col halves).
    alignas(64) float S_scratch[M_H * 16 * 2 * 2];  // 16 * 16 * 4 = 1024 floats
    // P scratch: the BF16-pair packed form of softmax(S), shape (M_Q/2, N_KV, 2)
    // i.e. one row per pair of Q rows -- exactly the layout the AMX tile load
    // expects for the second matmul.  Size: 16 * 64 * 2 = 2048 bf16 = 4 KiB.
    alignas(64) c10::BFloat16 P_pack[M_Q * N_KV];  // 32 * 32 = 1024 bf16 = 2 KiB

    // sm_scale broadcast.
    const __m512 v_scale = _mm512_set1_ps(sm_scale);

    // Strides (in bytes) for AMX tile loads.
    const int64_t stride_Q   = D     * sizeof(c10::BFloat16);   // row stride of Q
    const int64_t stride_Kp  = S_kv  * 2 * sizeof(c10::BFloat16); // row of packed K
    const int64_t stride_V   = D     * 2 * sizeof(c10::BFloat16); // row of packed V
    const int64_t stride_S   = 16    * sizeof(float);           // tile-local
    const int64_t stride_P   = N_KV  * sizeof(c10::BFloat16);   // P_pack stride

    // -------------------------------------------------------------------
    // Inner loop over K/V tiles of width N_KV.
    // -------------------------------------------------------------------
    const int64_t kv_tiles = S_kv / N_KV;
    const int64_t q_end    = q_start + M_Q;  // exclusive

    for (int64_t kvt = 0; kvt < kv_tiles; ++kvt) {
        const int64_t k_start = kvt * N_KV;
        const int64_t k_stop  = k_start + N_KV;

        // Causal early-out: if the entire K tile is past the last Q row,
        // skip it.  q_end is exclusive so the "last Q index + 1" is q_end.
        if (is_causal && k_start >= q_end) {
            break;
        }
        const bool needs_mask = is_causal && (k_stop > q_start);

        // -----------------------------------------------------------------
        // STEP 1+2+3: S = Q @ K^T (BF16 -> FP32) via four AMX matmuls.
        //
        // S shape is (M_Q=32, N_KV=32).  Each AMX matmul produces 16x16, so:
        //   (qh in 0..1)  (kh in 0..1):
        //       S[qh*16:(qh+1)*16, kh*16:(kh+1)*16] =
        //           sum_{d_chunk} Q[qh*16:(qh+1)*16, d_chunk] @
        //                         K^T[d_chunk, kh*16:(kh+1)*16]
        //
        // For D=64  we have 2 d_chunks of 32 bf16 each.
        // For D=128 we have 4 d_chunks of 32 bf16 each.
        // -----------------------------------------------------------------
        const int64_t d_chunks = D / 32;

        for (int qh = 0; qh < 2; ++qh) {        // 2 halves of Q rows
            for (int kh = 0; kh < 2; ++kh) {    // 2 halves of K cols (16 S each)
                _tile_zero(TMM_S);
                for (int dc = 0; dc < d_chunks; ++dc) {
                    // Q: row 16-block (qh), col 32-bf16-block (dc).
                    const c10::BFloat16* Qp = Q_tile_ptr
                        + (qh * M_H) * D
                        + dc * 32;

                    // K_pk: D-pair-block (dc covers 16 packed rows = 32 D),
                    // S column offset = (k_start + kh*16) columns * 2 bf16-pair
                    // stride.  k_start advances per outer KV-tile and was the
                    // missing piece -- without it every KV iteration would read
                    // the same first 16 S positions and the cross-tile online
                    // softmax would see all-zero scores for kvt>0.
                    const c10::BFloat16* Kp = K_pk_ptr
                        + (dc * 16) * S_kv * 2
                        + (k_start + kh * M_H) * 2;

                    _tile_loadd(TMM_Q, Qp, stride_Q);
                    _tile_loadd(TMM_K, Kp, stride_Kp);
                    _tile_dpbf16ps(TMM_S, TMM_Q, TMM_K);
                }
                // STEP 4: store the 16x16 fp32 tile to scratch.
                _tile_stored(TMM_S,
                             S_scratch + (qh * 2 + kh) * (M_H * 16),
                             stride_S);
            }
        }

        // -----------------------------------------------------------------
        // STEP 5: online softmax in AVX-512.
        //
        // We process the 32x32 S block row-by-row (two zmms per row -- the
        // first holding cols 0..15, the second cols 16..31).  Rescale O_acc,
        // update row_max and row_sum, and stash the unnormalised P values
        // for the second matmul.
        // -----------------------------------------------------------------
        alignas(64) float P_fp32[M_Q * N_KV];

        for (int i = 0; i < M_Q; ++i) {
            const int qh = i / M_H;
            const int qr = i % M_H;

            // Gather the two halves: scratch tile (qh, 0) and (qh, 1), row qr.
            __m512 s_lo = _mm512_load_ps(
                S_scratch + (qh * 2 + 0) * (M_H * 16) + qr * 16);
            __m512 s_hi = _mm512_load_ps(
                S_scratch + (qh * 2 + 1) * (M_H * 16) + qr * 16);

            s_lo = _mm512_mul_ps(s_lo, v_scale);
            s_hi = _mm512_mul_ps(s_hi, v_scale);

            // Causal mask: for this Q row at global index (q_start + i),
            // any K index > (q_start + i) must be set to -inf.
            if (needs_mask) {
                const int64_t q_idx = q_start + i;
                // Build column index vectors for the two halves.
                const __m512i idx_lo = _mm512_add_epi32(
                    _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
                    _mm512_set1_epi32(static_cast<int>(k_start)));
                const __m512i idx_hi = _mm512_add_epi32(
                    _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
                    _mm512_set1_epi32(static_cast<int>(k_start + 16)));
                const __m512i v_q    = _mm512_set1_epi32(static_cast<int>(q_idx));
                const __mmask16 m_lo = _mm512_cmpgt_epi32_mask(idx_lo, v_q);
                const __mmask16 m_hi = _mm512_cmpgt_epi32_mask(idx_hi, v_q);
                const __m512 neg_inf = _mm512_set1_ps(
                    -std::numeric_limits<float>::infinity());
                s_lo = _mm512_mask_blend_ps(m_lo, s_lo, neg_inf);
                s_hi = _mm512_mask_blend_ps(m_hi, s_hi, neg_inf);
            }

            // rowmax over 32 lanes.
            const float local_max = std::max(reduce_max_ps(s_lo),
                                              reduce_max_ps(s_hi));
            const float m_old = row_max[i];
            const float m_new = std::max(m_old, local_max);
            const float alpha = std::exp(m_old - m_new);  // safe even for -inf

            // p = exp(s - m_new)
            const __m512 v_mnew = _mm512_set1_ps(m_new);
            __m512 p_lo = exp_ps(_mm512_sub_ps(s_lo, v_mnew));
            __m512 p_hi = exp_ps(_mm512_sub_ps(s_hi, v_mnew));

            // Stash fp32 P for the cast step below.
            _mm512_store_ps(P_fp32 + i * N_KV + 0,  p_lo);
            _mm512_store_ps(P_fp32 + i * N_KV + 16, p_hi);

            const float l_local = reduce_add_ps(p_lo) + reduce_add_ps(p_hi);
            const float l_old   = row_sum[i];

            row_max[i] = m_new;
            row_sum[i] = alpha * l_old + l_local;

            // Rescale O_acc[i, :] by alpha.
            const __m512 v_alpha = _mm512_set1_ps(alpha);
            for (int64_t d = 0; d < D; d += 16) {
                __m512 o = _mm512_load_ps(O_acc + i * D + d);
                o = _mm512_mul_ps(o, v_alpha);
                _mm512_store_ps(O_acc + i * D + d, o);
            }
        }

        // -----------------------------------------------------------------
        // STEP 6: pack P into the BF16-pair layout that AMX wants.
        //
        // P_pack shape: (M_Q/2, N_KV, 2)  i.e. 16 rows of 64 bf16 = 1 KiB.
        // For each pair (2r, 2r+1) of Q rows, interleave their N_KV values:
        //     P_pack[r, k, 0] = P_fp32[2r,   k]  (cast to bf16)
        //     P_pack[r, k, 1] = P_fp32[2r+1, k]  (cast to bf16)
        //
        // This is the exact A-matrix layout (since for the second matmul,
        // P is the "A" operand and V the "B" operand, *both* need their
        // K-axis in BF16-pair form -- A pairs come from interleaving rows).
        //
        // Actually for the "A" operand AMX takes the natural row-major BF16
        // layout (16 rows x 32 bf16, K-dim along columns).  So pairing is
        // only needed on the "B" side -- which is V (already packed).
        // For "A" we just store P row by row.  The 32 K columns become the
        // K-dim of the matmul.
        // -----------------------------------------------------------------
        for (int i = 0; i < M_Q; ++i) {
            __m512 p_lo = _mm512_load_ps(P_fp32 + i * N_KV + 0);
            __m512 p_hi = _mm512_load_ps(P_fp32 + i * N_KV + 16);
            __m512i bf  = cvt_fp32x32_to_bf16i(p_lo, p_hi);
            _mm512_storeu_si512(
                reinterpret_cast<__m512i*>(P_pack + i * N_KV), bf);
        }

        // -----------------------------------------------------------------
        // STEP 7: O += P @ V  via AMX.
        //
        // O shape per matmul tile: 16 x 16 fp32.  We need full (32, D).
        //   - 2 Q halves (qh = 0,1) drive M_H=16 rows each.
        //   - D / 16 column tiles drive 16 head-dim lanes each.
        //   - Single K-dim chunk of 32 covers the whole N_KV in one
        //     _tile_dpbf16ps (since N_KV = 32 == bf16 K-chunk per AMX call).
        //
        // The V operand row stride is `stride_V = D * 4 B`  (each packed row
        // covers 2 logical K rows x D feature dims, BF16-pair).  We have
        // N_KV / 2 = 16 packed rows -- exactly one AMX tile worth.
        // -----------------------------------------------------------------
        const int64_t d_tiles = D / 16;
        const c10::BFloat16* V_base = V_p_ptr + (k_start / 2) * D * 2;

        for (int qh = 0; qh < 2; ++qh) {
            for (int64_t dt = 0; dt < d_tiles; ++dt) {
                // Load O sub-tile from O_acc into TMM_O so we can accumulate.
                _tile_loadd(TMM_O,
                            O_acc + (qh * M_H) * D + dt * 16,
                            D * sizeof(float));

                const c10::BFloat16* Pp = P_pack + (qh * M_H) * N_KV;
                // V slice for these 16 feature-dim columns.
                const c10::BFloat16* Vp = V_base + (dt * 16) * 2;

                _tile_loadd(TMM_P, Pp, stride_P);
                _tile_loadd(TMM_V, Vp, stride_V);
                _tile_dpbf16ps(TMM_O, TMM_P, TMM_V);

                _tile_stored(TMM_O,
                             O_acc + (qh * M_H) * D + dt * 16,
                             D * sizeof(float));
            }
        }
    }  // for kvt

    // -----------------------------------------------------------------
    // Final pass: O /= rowsum; cast to BF16; store.
    // -----------------------------------------------------------------
    for (int i = 0; i < M_Q; ++i) {
        const float inv_l = (row_sum[i] > 0.0f) ? (1.0f / row_sum[i]) : 0.0f;
        const __m512 v_inv = _mm512_set1_ps(inv_l);
        for (int64_t d = 0; d < D; d += 32) {
            __m512 o_lo = _mm512_load_ps(O_acc + i * D + d);
            __m512 o_hi = _mm512_load_ps(O_acc + i * D + d + 16);
            o_lo = _mm512_mul_ps(o_lo, v_inv);
            o_hi = _mm512_mul_ps(o_hi, v_inv);
            __m512i bf = cvt_fp32x32_to_bf16i(o_lo, o_hi);
            _mm512_storeu_si512(
                reinterpret_cast<__m512i*>(O_tile_ptr + i * D + d), bf);
        }
    }
}

}  // namespace


// =========================================================================
// PyBind entry point.
// =========================================================================
at::Tensor flash_attn_bf16(const at::Tensor& Q,
                           const at::Tensor& K,
                           const at::Tensor& V,
                           double sm_scale,
                           bool is_causal) {
    // ---- Shape/dtype/contiguity asserts.
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "Q/K/V must be 4D (B, H, S, D)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q/K/V shapes must match for this kernel");
    TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "Q must be bf16");
    TORCH_CHECK(K.scalar_type() == at::kBFloat16, "K must be bf16");
    TORCH_CHECK(V.scalar_type() == at::kBFloat16, "V must be bf16");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q/K/V must be contiguous");
    TORCH_CHECK(Q.device().is_cpu(), "this is the CPU/AMX kernel");

    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S = Q.size(2);
    const int64_t D = Q.size(3);

    TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128, got ", D);
    TORCH_CHECK(S >= 32 && S % 32 == 0,
                "sequence length must be a positive multiple of 32, got ", S);

    // ---- One-shot AMX permission opt-in.
    TORCH_CHECK(enable_amx_once(),
                "Linux refused arch_prctl(ARCH_REQ_XCOMP_PERM, "
                "XFEATURE_XTILEDATA). Kernel >= 5.16 with AMX support is "
                "required.");

    // ---- Pre-pack K and V into AMX B-layouts once per call (amortised over
    // S^2 work).
    at::Tensor K_pk = pack_K_for_amx_B(K);
    at::Tensor V_p  = pack_V_pairs(V);

    at::Tensor O = at::empty_like(Q);

    const auto* Q_ptr  = reinterpret_cast<const c10::BFloat16*>(Q.data_ptr());
    const auto* Kp_ptr = reinterpret_cast<const c10::BFloat16*>(K_pk.data_ptr());
    const auto* Vp_ptr = reinterpret_cast<const c10::BFloat16*>(V_p.data_ptr());
    auto*       O_ptr  = reinterpret_cast<c10::BFloat16*>(O.data_ptr());

    const int64_t M_Q       = 32;
    const int64_t q_tiles   = S / M_Q;
    const float   sm_scale_f = static_cast<float>(sm_scale);

    // ---- Outer parallel loop over (B, H, q_tile).  Each thread configures
    // its own AMX tiles, then walks its share of the iteration space.
    #pragma omp parallel
    {
        configure_tiles_for_thread();

        #pragma omp for collapse(3) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t qt = 0; qt < q_tiles; ++qt) {
                    const int64_t q_start = qt * M_Q;
                    const c10::BFloat16* Q_tile =
                        Q_ptr + ((b * H + h) * S + q_start) * D;
                    const c10::BFloat16* K_pk_tile =
                        Kp_ptr + (b * H + h) * (D / 2) * S * 2;
                    const c10::BFloat16* V_p_tile =
                        Vp_ptr + (b * H + h) * (S / 2) * D * 2;
                    c10::BFloat16* O_tile =
                        O_ptr + ((b * H + h) * S + q_start) * D;

                    process_q_tile(Q_tile, K_pk_tile, V_p_tile, O_tile,
                                   /*S_kv=*/S, /*D=*/D,
                                   q_start, sm_scale_f, is_causal);
                }
            }
        }

        _tile_release();
    }

    return O;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_bf16", &flash_attn_bf16,
          "Flash attention on AMX BF16");
}
