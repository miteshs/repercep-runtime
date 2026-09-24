// =============================================================================
//  Repercep CPU sibling kernels: flash-attention on Intel AMX INT8
// =============================================================================
//
//  This file implements a FlashAttention-2 style scaled-dot-product-attention
//  forward pass that targets Intel Sapphire Rapids+ via the AMX INT8 tile
//  instructions (TDPBSSD), AVX-512 BF16/FP32, and AVX-512F.
//
//  It is the INT8 sibling of the BF16 kernel in
//  `kernels/cpu/amx_attn/flash_attn_amx.cpp`.  The two kernels share the
//  outer FlashAttention-2 loop, the AMX tile-config plumbing, the AVX-512
//  online-softmax math, and the BF16-in / BF16-out *surface* contract;
//  they differ in the inner matmul (TDPBSSD vs TDPBF16PS), the AMX B-operand
//  packing (INT8 packs 64 lanes per tile row vs BF16's 32), and the extra
//  dynamic-quantization passes that fold each Q-tile's Q / K / V slabs from
//  BF16 down to INT8 with per-row FP32 scales before the matmul.
//
//  -------------------------------------------------------------------------
//  Algorithm at a glance
//  -------------------------------------------------------------------------
//
//    For each (batch, head) and Q-tile of M_Q=32 rows:
//        m_i  = -inf                                # running rowmax
//        l_i  =  0                                  # running rowsum
//        O_i  =  0                                  # running output acc
//        dyn-quantize Q_tile           -> Q_i8 (M_Q, D)   + q_scale[M_Q]
//        for each K-tile of N_KV=32 columns (early-out on causal):
//            dyn-quantize K_tile       -> K_i8 (N_KV, D)  + k_scale[N_KV]
//            dyn-quantize V_tile       -> V_i8 (N_KV, D)  + v_scale[D]
//            S_i32 = Q_i8 @ K_i8^T                  # AMX INT8 -> INT32
//            S     = S_i32 * (q_scale[i] * k_scale[j])  # broadcast outer
//            S    *= sm_scale
//            (optional causal mask)
//            m_new = max(m_i, rowmax(S))
//            alpha = exp(m_i - m_new)
//            P     = exp(S   - m_new)
//            l_i   = alpha * l_i  + rowsum(P)
//            dyn-quantize P            -> P_i8 (M_Q, N_KV) + p_scale[M_Q]
//            T_i32 = P_i8 @ V_i8                    # AMX INT8 -> INT32
//            O_i  = alpha * O_i + T_i32 * (p_scale[i] * v_scale[d])
//            m_i   = m_new
//        O_i /= l_i
//        cast O_i to BF16, store.
//
//  -------------------------------------------------------------------------
//  AMX tile layout (palette = 1)
//  -------------------------------------------------------------------------
//
//    Each AMX tile may carry up to 16 rows and 64 bytes/row (= 1 KiB).
//    For TDPBSSD, INT8 inputs are packed as 16 rows x (64 int8) = 64 B/row,
//    and the INT32 accumulator is 16 rows x (16 int32) = 64 B/row.
//
//    We use six tiles per inner step:
//        tmm0 := Q sub-tile   (16 x 64 i8,  row stride = D bytes)
//        tmm1 := K^T sub-tile (16 x 64 i8,  K-axis is the 4-wide lane group)
//        tmm2 := S accum      (16 x 16 i32, row stride = 16 * 4 = 64 B)
//        tmm3 := P sub-tile   (16 x 64 i8,  K-axis pre-packed in 4-byte lanes)
//        tmm4 := V sub-tile   (16 x 64 i8,  K-axis pre-packed in 4-byte lanes)
//        tmm5 := O accum      (16 x 16 i32, one head_dim col group)
//
//    Because a single AMX tile only covers 16 query rows but the spec asks
//    for M_Q=32 queries per Q-tile, we stripe each Q-tile into TWO 16-row
//    halves and run the inner sequence twice per K-tile.  This matches the
//    BF16 sibling exactly.
//
//  -------------------------------------------------------------------------
//  Quantization scheme (durable, in-source so it survives grep)
//  -------------------------------------------------------------------------
//
//    Three of the four tensors are quantized PER-ROW SYMMETRIC INT8 with an
//    FP32 scale (Q, K, P); V is quantized with a single PER-TILE symmetric
//    FP32 scale.  "Row" here means the K-axis-contiguous slab that is the
//    natural inner-product unit:
//        * Q: per-row of M_Q (one scale per query token)
//        * K: per-row of N_KV (one scale per key token)
//        * P: per-row of M_Q (one scale per query token, recomputed per
//             K-tile because P's row magnitudes change with the online
//             softmax)
//        * V: ONE scale per K-tile (per-row would put v_scale inside the
//             P @ V sum and we'd lose the ability to dequant cheaply --
//             see the long comment above the PV step in process_q_tile).
//
//    The per-row scheme matches the per-token-symmetric scheme used by
//    oneDNN's IPEX smooth-quant and by the per-row scheme in
//    `src/repercep/runtime/quantize.py`, so the numerics are coherent with
//    the rest of the runtime.  V's coarser per-tile scale costs a little
//    range but is mathematically clean; revisit per-row + per-channel
//    scale absorption when we move past correctness-first.
//
//    Score-tile dequant:    S_fp32[i, j] = (q_scale[i] * k_scale[j]) * S_i32
//    PV-output dequant:     O_fp32[i, :] += (p_scale[i] * v_tile_scale) * T_i32
//
//  -------------------------------------------------------------------------
//  AMX INT8 B-operand layout (durable, in-source so it survives grep)
//  -------------------------------------------------------------------------
//
//    TDPBSSD reads its B operand as 16 rows, each row holding 64 INT8 lanes,
//    arranged so that the K-axis advances 4 entries per row-step and 0 per
//    column-step.  Concretely, for a logical (K_in, N_out) matrix B, the
//    AMX-friendly layout is (K_in/4, N_out, 4) with the innermost 4 holding
//    consecutive K-axis values.
//
//    For Q @ K^T we want B := K^T of shape (D, N_KV), so the packed layout
//    is (D/4, N_KV, 4) with each inner 4 carrying 4 consecutive head-dim
//    components of one K row.
//
//    For P @ V we want B := V of shape (N_KV, D), so the packed layout is
//    (N_KV/4, D, 4) with each inner 4 carrying 4 consecutive K rows of one
//    feature dim.
//
//  -------------------------------------------------------------------------
//  Optimisation notes (durable)
//  -------------------------------------------------------------------------
//
//    * Quantization is dynamic and per-tile.  Static weight quant lives in
//      `src/repercep/runtime/quantize.py` and is the right path for Linear
//      layers; here the activation rows change every step and there is no
//      offline "weight" to pre-quantize -- Q, K, and V are all activations.
//    * Per-tile dequant scales are broadcast through the inner loops with
//      a row-vs-col split (q_scale broadcast down rows; k_scale broadcast
//      across columns).  The two are folded into a single fp32 fma in the
//      softmax pass; cost is amortised in front of the AMX matmul.
//    * Per the throughput note in F23, TDPBSSD is 2x TDPBF16PS on SPR.  The
//      dynamic-quant overhead currently eats most of that for short K
//      tiles; long-context (S >= 4096) benchmarking is the right place to
//      validate the speedup.  Correctness first; the optimization is
//      hardware-blocked on this VM anyway (no AMX exposed).
//    * `_tile_release()` is called at the end of each parallel region so
//      libraries that touch AMX state after the kernel returns (e.g.
//      oneDNN) see a clean tile-config slot.
//    * On `is_causal`, we skip K-tiles whose first K index is strictly past
//      the last Q index in the current Q-tile.  Inside the boundary tile
//      we apply a fine-grained -inf mask in AVX-512 (post-dequant).
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

// Tile assignments used throughout the kernel.  See the BF16 sibling for the
// rationale on why these MUST be preprocessor macros (gcc's intrinsic macros
// stringize the tile number argument into inline-asm).
#define TMM_Q 0   // 16 x 64 int8  (Q sub-tile)
#define TMM_K 1   // 16 x 64 int8  (K^T sub-tile -- 64 D-bytes packed per row)
#define TMM_S 2   // 16 x 16 int32 (S = Q @ K^T accumulator)
#define TMM_P 3   // 16 x 64 int8  (P sub-tile fed to second matmul)
#define TMM_V 4   // 16 x 64 int8  (V sub-tile -- INT8 4-lane packed)
#define TMM_O 5   // 16 x 16 int32 (O sub-accumulator per head_dim col group)

// Set up the tile config for a single thread.  Must be called once per thread
// before any AMX instruction.  We configure all 6 tiles up front; unused
// tiles cost nothing.
void configure_tiles_for_thread() {
    TileConfig cfg = {};
    cfg.palette_id = 1;
    cfg.start_row  = 0;

    // INT8 inputs: 16 rows x 64 int8 = 16 x 64 B
    cfg.rows[TMM_Q] = 16;  cfg.colsb[TMM_Q] = 64;
    cfg.rows[TMM_K] = 16;  cfg.colsb[TMM_K] = 64;
    cfg.rows[TMM_P] = 16;  cfg.colsb[TMM_P] = 64;
    cfg.rows[TMM_V] = 16;  cfg.colsb[TMM_V] = 64;

    // INT32 accumulators: 16 rows x 16 int32 = 16 x 64 B
    cfg.rows[TMM_S] = 16;  cfg.colsb[TMM_S] = 64;
    cfg.rows[TMM_O] = 16;  cfg.colsb[TMM_O] = 64;

    _tile_loadconfig(&cfg);
}

// ---------------------------------------------------------------------------
// Helpers: BF16 <-> FP32 casts in AVX-512.  We rely on -mavx512bf16.
// ---------------------------------------------------------------------------

// Convert 16 BF16 lanes (lower 256 bits of a 512-bit reg) to 16 FP32 lanes.
inline __m512 cvt_bf16x16_to_fp32(__m256i bf16x16) {
    __m512i as_i32 = _mm512_cvtepu16_epi32(bf16x16);
    as_i32 = _mm512_slli_epi32(as_i32, 16);
    return _mm512_castsi512_ps(as_i32);
}

// Convert 32 FP32 lanes (two zmms) -> 32 BF16 lanes as raw __m512i with RNE.
inline __m512i cvt_fp32x32_to_bf16i(__m512 a_lo, __m512 b_hi) {
    __m512bh bf = _mm512_cvtne2ps_pbh(b_hi, a_lo);
    __m512i out;
    std::memcpy(&out, &bf, sizeof(out));
    return out;
}

// Horizontal max / add across 16 fp32 lanes.
inline float reduce_max_ps(__m512 v) { return _mm512_reduce_max_ps(v); }
inline float reduce_add_ps(__m512 v) { return _mm512_reduce_add_ps(v); }

// Horizontal max-abs across 16 fp32 lanes (returns a non-negative float).
inline float reduce_max_abs_ps(__m512 v) {
    const __m512 absmask = _mm512_castsi512_ps(
        _mm512_set1_epi32(0x7fffffff));
    return _mm512_reduce_max_ps(_mm512_and_ps(v, absmask));
}

// Vectorised exp() approximation -- same as the BF16 sibling, see that file
// for the derivation.  The -inf clamp is critical for causal masking.
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

    __m512 y  = _mm512_mul_ps(x, LOG2E);
    __m512 n  = _mm512_roundscale_ps(y, _MM_FROUND_TO_NEAREST_INT |
                                        _MM_FROUND_NO_EXC);
    __m512 f  = _mm512_sub_ps(y, n);
    __m512 g  = _mm512_mul_ps(f, C1);

    __m512 p  = C5;
    p = _mm512_fmadd_ps(p, g, C4);
    p = _mm512_fmadd_ps(p, g, C3);
    p = _mm512_fmadd_ps(p, g, C2);
    p = _mm512_fmadd_ps(p, g, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, g, C0);
    p = _mm512_fmadd_ps(p, g, C0);

    __m512i ni = _mm512_cvtps_epi32(n);
    ni = _mm512_slli_epi32(_mm512_add_epi32(ni, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(ni));
}

// ---------------------------------------------------------------------------
// Per-row symmetric INT8 quant of a BF16 tile of shape (M_rows, D), producing
// an INT8 buffer of shape (M_rows, D) and an FP32 scale vector of length
// M_rows.  ``D`` must be a multiple of 16.
//
// The output INT8 buffer is in *row-major natural* layout (not yet AMX
// B-packed) -- callers that feed it into the AMX A-operand can use it
// directly; callers that need the B-operand layout must repack via
// pack_int8_B_*.
// ---------------------------------------------------------------------------
void quantize_rows_bf16_to_int8(const c10::BFloat16* __restrict src,
                                int64_t M_rows,
                                int64_t D,
                                int8_t* __restrict dst,
                                float* __restrict scales) {
    for (int64_t i = 0; i < M_rows; ++i) {
        const c10::BFloat16* row = src + i * D;
        float row_absmax = 0.0f;

        // Pass 1: per-row max-abs in fp32.
        for (int64_t d = 0; d < D; d += 16) {
            __m256i bf  = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + d));
            __m512  fp  = cvt_bf16x16_to_fp32(bf);
            const float local = reduce_max_abs_ps(fp);
            if (local > row_absmax) row_absmax = local;
        }

        // Scale: symmetric, range [-127, 127] (we leave -128 on the table
        // to match the per-channel weight quant in runtime/quantize.py).
        const float scale     = (row_absmax > 0.0f) ? (row_absmax / 127.0f)
                                                    : 1.0f;
        const float inv_scale = 1.0f / scale;
        scales[i] = scale;

        const __m512 v_inv = _mm512_set1_ps(inv_scale);

        // Pass 2: quantize.  Round-to-nearest-even, saturate to int8.
        for (int64_t d = 0; d < D; d += 16) {
            __m256i bf  = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + d));
            __m512  fp  = cvt_bf16x16_to_fp32(bf);
            fp = _mm512_mul_ps(fp, v_inv);
            // Round to nearest even, then convert to int32 with saturation
            // via cvtepi32 + min/max clamp before narrowing to int8.
            __m512i  i32 = _mm512_cvtps_epi32(fp);
            i32 = _mm512_max_epi32(i32, _mm512_set1_epi32(-127));
            i32 = _mm512_min_epi32(i32, _mm512_set1_epi32( 127));
            // Narrow 16 x int32 -> 16 x int8 (signed saturation).
            __m128i i8  = _mm512_cvtsepi32_epi8(i32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i * D + d), i8);
        }
    }
}

// Same as above but the input is FP32 (used for the P tile after softmax).
void quantize_rows_fp32_to_int8(const float* __restrict src,
                                int64_t M_rows,
                                int64_t D,
                                int8_t* __restrict dst,
                                float* __restrict scales) {
    for (int64_t i = 0; i < M_rows; ++i) {
        const float* row = src + i * D;
        float row_absmax = 0.0f;

        for (int64_t d = 0; d < D; d += 16) {
            __m512 fp = _mm512_loadu_ps(row + d);
            const float local = reduce_max_abs_ps(fp);
            if (local > row_absmax) row_absmax = local;
        }

        const float scale     = (row_absmax > 0.0f) ? (row_absmax / 127.0f)
                                                    : 1.0f;
        const float inv_scale = 1.0f / scale;
        scales[i] = scale;
        const __m512 v_inv = _mm512_set1_ps(inv_scale);

        for (int64_t d = 0; d < D; d += 16) {
            __m512 fp = _mm512_loadu_ps(row + d);
            fp = _mm512_mul_ps(fp, v_inv);
            __m512i  i32 = _mm512_cvtps_epi32(fp);
            i32 = _mm512_max_epi32(i32, _mm512_set1_epi32(-127));
            i32 = _mm512_min_epi32(i32, _mm512_set1_epi32( 127));
            __m128i i8  = _mm512_cvtsepi32_epi8(i32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i * D + d), i8);
        }
    }
}

// ---------------------------------------------------------------------------
// Pack an INT8 (K_in, N_out) matrix into AMX B-operand layout
// (K_in/4, N_out, 4).  K_in must be a multiple of 4.
//
// For Q @ K^T:  B := K^T of shape (D, N_KV).  This packs the head-dim axis
//               into 4-lane groups so each AMX tile load reads 64 int8 / row.
// For P @ V:    B := V   of shape (N_KV, D).   This packs the K-axis into
//               4-lane groups; one packed row covers 4 K-rows worth of one
//               feature column.
// ---------------------------------------------------------------------------
void pack_int8_B(const int8_t* __restrict src,
                 int64_t K_in,
                 int64_t N_out,
                 int8_t* __restrict dst) {
    // src logical layout: row-major (K_in, N_out)
    // dst layout:         (K_in/4, N_out, 4)
    for (int64_t kg = 0; kg < K_in / 4; ++kg) {
        for (int64_t n = 0; n < N_out; ++n) {
            int8_t* slot = dst + (kg * N_out + n) * 4;
            slot[0] = src[(4 * kg + 0) * N_out + n];
            slot[1] = src[(4 * kg + 1) * N_out + n];
            slot[2] = src[(4 * kg + 2) * N_out + n];
            slot[3] = src[(4 * kg + 3) * N_out + n];
        }
    }
}

// =========================================================================
// Core per-Q-tile worker.
//
// Each call processes ONE 32-row Q tile against the entire K dimension.
// Inputs are BF16; outputs are BF16; all INT8 quantization happens inside
// this function on per-tile scratch buffers.
//
// Args:
//   Q_tile_ptr   : (M_Q=32, D)    bf16   -- one Q tile
//   K_full_ptr   : (S_kv, D)      bf16   -- full K for this (B, H)
//   V_full_ptr   : (S_kv, D)      bf16   -- full V for this (B, H)
//   O_tile_ptr   : (M_Q=32, D)    bf16 out
//   S_kv         : number of K tokens
//   D            : head_dim, must be 64 or 128
//   q_start      : starting Q index in the global sequence (for causal mask)
//   sm_scale     : scalar pre-softmax multiplier
//   is_causal    : enable causal masking
// =========================================================================
void process_q_tile(const c10::BFloat16* __restrict Q_tile_ptr,
                    const c10::BFloat16* __restrict K_full_ptr,
                    const c10::BFloat16* __restrict V_full_ptr,
                    c10::BFloat16* __restrict       O_tile_ptr,
                    int64_t S_kv,
                    int64_t D,
                    int64_t q_start,
                    float sm_scale,
                    bool is_causal) {
    constexpr int M_Q  = 32;   // queries per tile  (= 2 AMX tile halves)
    constexpr int N_KV = 32;   // keys per inner step (= 2 AMX tile col halves)
    constexpr int M_H  = 16;   // AMX hard-coded tile row count

    // ---- Per-row online softmax state, one fp32 entry per Q row.
    alignas(64) float row_max[M_Q];
    alignas(64) float row_sum[M_Q];
    for (int i = 0; i < M_Q; ++i) {
        row_max[i] = -std::numeric_limits<float>::infinity();
        row_sum[i] = 0.0f;
    }

    // ---- O accumulator in FP32, full (M_Q, D).  Max footprint at D=128 is
    // 32*128*4 = 16 KiB which comfortably stays in L1d.
    alignas(64) float O_acc[M_Q * 128];
    std::memset(O_acc, 0, sizeof(float) * M_Q * D);

    // ---- Per-tile INT8 scratch buffers.
    //
    // Q_i8:        (M_Q, D)            up to 32 * 128 = 4 KiB
    // q_scales:    (M_Q,)              128 B
    // Kt_i8:       (N_KV, D)           up to 32 * 128 = 4 KiB
    // Kt_packed:   (D/4, N_KV, 4)      same byte count, AMX-B layout
    // k_scales:    (N_KV,)             128 B
    // V_i8:        (N_KV, D)           up to 32 * 128 = 4 KiB
    // V_packed:    (N_KV/4, D, 4)      same byte count, AMX-B layout
    // P_i8:        (M_Q, N_KV)         1 KiB
    // p_scales:    (M_Q,)              128 B
    // S_scratch:   (M_Q, N_KV) int32   4 KiB (raw AMX accumulator output)
    // O_scratch:   (M_Q, D)    int32   up to 16 KiB
    //
    // Total upper bound at D=128: ~50 KiB per thread, comfortably under
    // L2 on Sapphire Rapids (2 MiB/core).
    alignas(64) int8_t  Q_i8     [M_Q  * 128];
    alignas(64) float   q_scales [M_Q];
    alignas(64) int8_t  Kt_i8    [N_KV * 128];
    alignas(64) int8_t  Kt_packed[N_KV * 128];
    alignas(64) float   k_scales [N_KV];
    alignas(64) int8_t  V_i8     [N_KV * 128];
    alignas(64) int8_t  V_packed [N_KV * 128];
    alignas(64) int8_t  P_i8     [M_Q  * N_KV];
    alignas(64) float   p_scales [M_Q];
    alignas(64) int32_t S_scratch[M_Q  * N_KV];
    alignas(64) int32_t O_scratch[M_Q  * 128];
    alignas(64) float   P_fp32   [M_Q  * N_KV];

    // -------------------------------------------------------------------
    // Quantize Q once for the entire Q tile (Q does not change across
    // K-tiles within this call).
    // -------------------------------------------------------------------
    quantize_rows_bf16_to_int8(Q_tile_ptr, M_Q, D, Q_i8, q_scales);

    const int64_t kv_tiles = S_kv / N_KV;
    const int64_t q_end    = q_start + M_Q;  // exclusive

    // Tile-load strides (in bytes).
    const int64_t stride_Q_i8 = D     * sizeof(int8_t);   // Q_i8 row stride
    const int64_t stride_Kp   = N_KV  * 4;                // packed K row stride
    const int64_t stride_S_i  = N_KV  * sizeof(int32_t);  // S_scratch stride
    const int64_t stride_O_i  = D     * sizeof(int32_t);  // O_scratch stride

    for (int64_t kvt = 0; kvt < kv_tiles; ++kvt) {
        const int64_t k_start = kvt * N_KV;
        const int64_t k_stop  = k_start + N_KV;

        if (is_causal && k_start >= q_end) break;
        const bool needs_mask = is_causal && (k_stop > q_start);

        // -----------------------------------------------------------------
        // STEP 1: dynamic-quantize this K and V tile.
        //   K: per-row symmetric INT8 (one scale per K token).
        //   V: per-tile symmetric INT8 (one scalar scale shared across
        //      every K row of this tile; see rationale above the PV step).
        // -----------------------------------------------------------------
        const c10::BFloat16* K_tile = K_full_ptr + k_start * D;
        const c10::BFloat16* V_tile = V_full_ptr + k_start * D;

        quantize_rows_bf16_to_int8(K_tile, N_KV, D, Kt_i8, k_scales);

        // V per-tile scale: max-abs over the entire (N_KV, D) slab.
        float v_tile_absmax = 0.0f;
        for (int n = 0; n < N_KV; ++n) {
            const c10::BFloat16* row = V_tile + n * D;
            for (int64_t d = 0; d < D; d += 16) {
                __m256i bf = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row + d));
                __m512  fp = cvt_bf16x16_to_fp32(bf);
                const float local = reduce_max_abs_ps(fp);
                if (local > v_tile_absmax) v_tile_absmax = local;
            }
        }
        const float v_tile_scale = (v_tile_absmax > 0.0f)
                                   ? (v_tile_absmax / 127.0f) : 1.0f;
        const float v_inv_scale  = 1.0f / v_tile_scale;
        {
            const __m512 v_inv = _mm512_set1_ps(v_inv_scale);
            for (int n = 0; n < N_KV; ++n) {
                const c10::BFloat16* row = V_tile + n * D;
                for (int64_t d = 0; d < D; d += 16) {
                    __m256i bf  = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(row + d));
                    __m512  fp  = cvt_bf16x16_to_fp32(bf);
                    fp = _mm512_mul_ps(fp, v_inv);
                    __m512i i32 = _mm512_cvtps_epi32(fp);
                    i32 = _mm512_max_epi32(i32, _mm512_set1_epi32(-127));
                    i32 = _mm512_min_epi32(i32, _mm512_set1_epi32( 127));
                    __m128i i8  = _mm512_cvtsepi32_epi8(i32);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(
                        V_i8 + n * D + d), i8);
                }
            }
        }

        // -----------------------------------------------------------------
        // STEP 2: pack K and V into AMX B-layouts.
        //
        // K: logical (N_KV, D) row-major; we want B := K^T of shape (D, N_KV)
        //    in (D/4, N_KV, 4) form.  We transpose during pack:
        //        dst[d/4, n, k] = K[n, 4*(d/4) + k].
        // V: logical (N_KV, D); we want B := V of shape (N_KV, D) in
        //    (N_KV/4, D, 4) form.  No transpose:
        //        dst[n/4, d, k] = V[4*(n/4) + k, d].
        // -----------------------------------------------------------------
        for (int64_t dg = 0; dg < D / 4; ++dg) {
            for (int64_t n = 0; n < N_KV; ++n) {
                int8_t* slot = Kt_packed + (dg * N_KV + n) * 4;
                slot[0] = Kt_i8[n * D + 4 * dg + 0];
                slot[1] = Kt_i8[n * D + 4 * dg + 1];
                slot[2] = Kt_i8[n * D + 4 * dg + 2];
                slot[3] = Kt_i8[n * D + 4 * dg + 3];
            }
        }
        pack_int8_B(V_i8, /*K_in=*/N_KV, /*N_out=*/D, V_packed);

        // -----------------------------------------------------------------
        // STEP 3: S_i32 = Q_i8 @ K_i8^T via AMX TDPBSSD.
        //
        // S shape: (M_Q=32, N_KV=32) of int32.  Each AMX matmul produces
        // 16x16 of int32 with K-dim chunk of 64 int8.  With D=64 that's
        // one d-chunk; with D=128 that's two d-chunks.
        //
        // We stripe over 2 Q-halves and 2 K-column-halves; matches BF16.
        // -----------------------------------------------------------------
        const int64_t d_chunks = D / 64;

        for (int qh = 0; qh < 2; ++qh) {
            for (int kh = 0; kh < 2; ++kh) {
                _tile_zero(TMM_S);
                for (int64_t dc = 0; dc < d_chunks; ++dc) {
                    // Q: row 16-block (qh), col 64-int8-block (dc).
                    const int8_t* Qp = Q_i8
                        + (qh * M_H) * D
                        + dc * 64;
                    // K_packed: D-quad-block (dc * 16 packed rows = 64 D),
                    // S column offset = (kh * 16) entries within this tile's
                    // N_KV slab.  Each packed K row is 4 bytes per N column.
                    const int8_t* Kp = Kt_packed
                        + (dc * 16) * N_KV * 4
                        + (kh * M_H) * 4;

                    _tile_loadd(TMM_Q, Qp, stride_Q_i8);
                    _tile_loadd(TMM_K, Kp, stride_Kp);
                    _tile_dpbssd(TMM_S, TMM_Q, TMM_K);
                }
                // Store 16x16 int32 tile into S_scratch at row qh*16, col kh*16.
                _tile_stored(TMM_S,
                             S_scratch + (qh * M_H) * N_KV + (kh * M_H),
                             stride_S_i);
            }
        }

        // -----------------------------------------------------------------
        // STEP 4: dequant + softmax in AVX-512.
        //
        // S_fp32[i, j] = q_scales[i] * k_scales[j] * S_i32[i, j] * sm_scale
        //
        // We process the 32x32 S block row-by-row (two zmms per row).  The
        // q-scale is constant within the row; the k-scale is per-column so
        // we load it as 16 floats per zmm.  Causal mask is applied after
        // dequant in fp32.
        // -----------------------------------------------------------------
        for (int i = 0; i < M_Q; ++i) {
            const float qs = q_scales[i] * sm_scale;
            const __m512 v_qs = _mm512_set1_ps(qs);

            // Two 16-wide halves: cols 0..15 and 16..31.
            __m512i s_lo_i = _mm512_load_si512(
                reinterpret_cast<const __m512i*>(S_scratch + i * N_KV + 0));
            __m512i s_hi_i = _mm512_load_si512(
                reinterpret_cast<const __m512i*>(S_scratch + i * N_KV + 16));
            __m512 s_lo = _mm512_cvtepi32_ps(s_lo_i);
            __m512 s_hi = _mm512_cvtepi32_ps(s_hi_i);

            __m512 ks_lo = _mm512_loadu_ps(k_scales + 0);
            __m512 ks_hi = _mm512_loadu_ps(k_scales + 16);

            s_lo = _mm512_mul_ps(s_lo, _mm512_mul_ps(v_qs, ks_lo));
            s_hi = _mm512_mul_ps(s_hi, _mm512_mul_ps(v_qs, ks_hi));

            // Causal mask: K indices > (q_start + i) become -inf.
            if (needs_mask) {
                const int64_t q_idx = q_start + i;
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

            const float local_max = std::max(reduce_max_ps(s_lo),
                                             reduce_max_ps(s_hi));
            const float m_old = row_max[i];
            const float m_new = std::max(m_old, local_max);
            const float alpha = std::exp(m_old - m_new);  // safe for -inf

            const __m512 v_mnew = _mm512_set1_ps(m_new);
            __m512 p_lo = exp_ps(_mm512_sub_ps(s_lo, v_mnew));
            __m512 p_hi = exp_ps(_mm512_sub_ps(s_hi, v_mnew));

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
        // STEP 5: dynamic-quantize P (per-row symmetric INT8) into P_i8.
        // Row magnitudes change every step because of the online softmax,
        // so this scale must be recomputed per (Q-tile x K-tile).
        // -----------------------------------------------------------------
        quantize_rows_fp32_to_int8(P_fp32, M_Q, N_KV, P_i8, p_scales);

        // -----------------------------------------------------------------
        // STEP 6: O_acc += dequant( P_i8 @ V_i8 ) via AMX TDPBSSD.
        //
        // TDPBSSD K-chunk per call = 64 int8.  Our K dim here is N_KV = 32,
        // which is HALF a chunk -- so the tile load needs zero-padding on
        // the K-axis to hit a full 64-lane chunk.  We allocate the
        // tile-load buffers (P_amx, V_amx) at the full chunk size and
        // zero-extend the upper halves; the zero columns contribute zero
        // to the int32 dot product and the AMX hardware swallows them at
        // line-rate.
        //
        // P A-operand: AMX wants A as 16 rows x 64 int8, K-axis along the
        //              64 cols.  P_i8 is already row-major (M_Q, N_KV) =
        //              (32, 32), so each 16-row half has 32 populated K
        //              cols + 32 zero-pad K cols.  P_amx shape: (M_Q, 64).
        //
        // V B-operand: packed V buffer is (N_KV/4, D, 4) = (8, D, 4).
        //              The AMX K-chunk wants 16 packed rows = 64 logical
        //              K positions.  V_amx shape: (16, D, 4); the upper
        //              (16 - N_KV/4) = 8 packed rows are zero.
        //
        // Why the V dequant scale is per-tile (one scalar) rather than
        // per-row:  P @ V's K-axis is N_KV, so for output col d we have
        //     T[i, d] = sum_n  P[i, n] * V[n, d]
        // If V had a per-row scale v_scales[n], that factor would live
        // INSIDE the sum and could not be pulled out before the matmul.
        // Folding v_scales into P would force a per-(i, n) rescale that
        // defeats the INT8 fast path.  Using one scalar V scale (computed
        // in STEP 1 above) lets us factor cleanly:
        //     T[i, d] = (p_scales[i] * v_tile_scale) * T_i32[i, d]
        // at the cost of a coarser representable range for V.  Per-row
        // V quant with per-(i, n) absorption is a future-work tunable.
        // -----------------------------------------------------------------

        alignas(64) int8_t P_amx[M_Q * 64];
        std::memset(P_amx, 0, sizeof(P_amx));
        for (int i = 0; i < M_Q; ++i) {
            std::memcpy(P_amx + i * 64, P_i8 + i * N_KV, N_KV);
        }

        alignas(64) int8_t V_amx[16 * 128 * 4];
        std::memset(V_amx, 0, sizeof(int8_t) * 16 * D * 4);
        std::memcpy(V_amx, V_packed, sizeof(int8_t) * (N_KV / 4) * D * 4);

        const int64_t stride_P_amx = 64 * sizeof(int8_t);
        const int64_t stride_V_amx = D  * 4 * sizeof(int8_t);

        const int64_t d_tiles = D / 16;

        for (int qh = 0; qh < 2; ++qh) {
            for (int64_t dt = 0; dt < d_tiles; ++dt) {
                _tile_zero(TMM_O);

                const int8_t* Pp = P_amx + (qh * M_H) * 64;
                const int8_t* Vp = V_amx + (dt * 16) * 4;
                _tile_loadd(TMM_P, Pp, stride_P_amx);
                _tile_loadd(TMM_V, Vp, stride_V_amx);
                _tile_dpbssd(TMM_O, TMM_P, TMM_V);

                _tile_stored(TMM_O,
                             O_scratch + (qh * M_H) * D + dt * 16,
                             stride_O_i);
            }
        }

        // STEP 7: dequant O_scratch and accumulate into O_acc.
        //   O_acc[i, d] += (p_scales[i] * v_tile_scale) * O_scratch[i, d]
        for (int i = 0; i < M_Q; ++i) {
            const float per_row = p_scales[i] * v_tile_scale;
            const __m512 v_pr = _mm512_set1_ps(per_row);
            for (int64_t d = 0; d < D; d += 16) {
                __m512i raw_i = _mm512_load_si512(
                    reinterpret_cast<const __m512i*>(O_scratch + i * D + d));
                __m512  raw   = _mm512_cvtepi32_ps(raw_i);
                __m512  prev  = _mm512_load_ps(O_acc + i * D + d);
                __m512  add   = _mm512_mul_ps(raw, v_pr);
                _mm512_store_ps(O_acc + i * D + d,
                                _mm512_add_ps(prev, add));
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
at::Tensor flash_attn_int8(const at::Tensor& Q,
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

    at::Tensor O = at::empty_like(Q);

    const auto* Q_ptr = reinterpret_cast<const c10::BFloat16*>(Q.data_ptr());
    const auto* K_ptr = reinterpret_cast<const c10::BFloat16*>(K.data_ptr());
    const auto* V_ptr = reinterpret_cast<const c10::BFloat16*>(V.data_ptr());
    auto*       O_ptr = reinterpret_cast<c10::BFloat16*>(O.data_ptr());

    const int64_t M_Q      = 32;
    const int64_t q_tiles  = S / M_Q;
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
                    const c10::BFloat16* K_full =
                        K_ptr + (b * H + h) * S * D;
                    const c10::BFloat16* V_full =
                        V_ptr + (b * H + h) * S * D;
                    c10::BFloat16* O_tile =
                        O_ptr + ((b * H + h) * S + q_start) * D;

                    process_q_tile(Q_tile, K_full, V_full, O_tile,
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
    m.def("flash_attn_int8", &flash_attn_int8,
          "Flash attention on AMX INT8 (TDPBSSD) -- BF16-in, BF16-out");
}
