// FP8 GEMM kernel for CDNA3 (gfx942) MI300X — toolchain proof-of-life.
//
// This is a thin, hand-written wrapper around the
// V_MFMA_F32_16x16x32_FP8_FP8 instruction — the gfx942 native FP8 MFMA tile.
//
// Status: correct, not fast.  It exists so the HIP toolchain is wired
// end-to-end (kernel -> pybind -> AttentionOp) and the operand layout is
// pinned by a test.  The perf path is kernels/triton_kernels/fp8_flash_attn.py,
// where Triton's codegen handles tile scheduling, LDS double-buffering and
// pipeline stages for us.  Do not promote this file without an LDS staging
// rewrite; hipBLASLt and the Triton kernel both beat it trivially.
//
// Operand register layout
// -----------------------
// The 2026-08 rewrite fixed the A/B fragment mapping, which previously loaded
// on only 16 of the 64 lanes and ignored the K-group term — so 3/4 of every
// K=32 contraction was silently zero.  Nothing called this kernel, so no
// published number was affected; see docs/BUILD_LOG.md.
//
// The mapping below is derived from the MFMA fragment geometry rather than
// guessed.  For an (M, N, K) = (16, 16, 32) FP8 tile on a wave64:
//
//   A is M x K = 16 x 32 = 512 elements over 64 lanes = 8 FP8 per lane.
//   B is K x N = 32 x 16 = 512 elements over 64 lanes = 8 FP8 per lane.
//   C is M x N = 16 x 16 = 256 floats over 64 lanes = 4 f32 per lane.
//
// The A/B fragment is distributed column-major over (M, WAVE / M), i.e. the
// lane's low 4 bits pick the row (A) or column (B) and the high 2 bits pick
// which contiguous group of 8 K-elements the lane owns:
//
//   A: m = lane % 16,  k = k_tile + (lane / 16) * 8 + e,  e in [0, 8)
//   B: n = lane % 16,  k = k_tile + (lane / 16) * 8 + e,  e in [0, 8)
//   C: col = lane % 16,  row = (lane / 16) * 4 + i,       i in [0, 4)
//
// Cross-checked against Modular's open-source MAX kernels (Apache-2.0), which
// derive the same mapping generically:
//   - the intrinsic is gated on per-lane fragment lengths (8, 8, 4, 4) in
//     max/mojo/max/gpu/compute/arch/mma_amd.mojo
//   - the A/B lane mapping is Layout.col_major(mma_m, WARP_SIZE // mma_m) in
//     max/kernels/src/layout/tensor_core.mojo::_load_a_amd
//   - the C mapping is Layout.row_major(mma_m // reg_per_thread, mma_n)
// tests/test_mfma_fragment_layout.py pins all three.
//
// References:
// - AMD CDNA3 ISA reference, "v_mfma_f32_16x16x32_fp8_fp8"
// - hip_fp8.h — gfx942 FP8 types (e4m3fnuz / e5m2fnuz).  CDNA3 is FNUZ;
//   CDNA4 (gfx950) switched to the OCP encodings.  A gfx950 port must
//   change the dtype, not just the arch flag.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <cstdint>

// gfx942 native MFMA tile shape: 16x16x32 with FP8 inputs, FP32 accumulator.
// One MFMA instruction does a 16x16 x 16x32 matmul over 32 contracted FP8
// elements.
constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 32;

// HIP wave size on gfx942 is 64.
constexpr int WAVE = 64;

// Per-lane fragment sizes, derived rather than hardcoded: the wave holds the
// whole tile, so each lane holds (tile elements / WAVE) of it.
constexpr int A_FRAG = (MFMA_M * MFMA_K) / WAVE;  // 8 FP8
constexpr int B_FRAG = (MFMA_K * MFMA_N) / WAVE;  // 8 FP8
constexpr int C_FRAG = (MFMA_M * MFMA_N) / WAVE;  // 4 f32

static_assert(A_FRAG == 8, "A fragment must be 8 FP8 per lane for 16x16x32");
static_assert(B_FRAG == 8, "B fragment must be 8 FP8 per lane for 16x16x32");
static_assert(C_FRAG == 4, "C fragment must be 4 f32 per lane for 16x16x32");

using fp8_t = __hip_fp8_storage_t;
using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// fp8 GEMM: C(MxN) = A(MxK) @ B(KxN), inputs row-major FP8, output BF16.
// One block (one wave) computes one MFMA_M x MFMA_N output tile.
__global__ void fp8_gemm_kernel(
    const fp8_t* __restrict__ A,     // (M, K) row-major
    const fp8_t* __restrict__ B,     // (K, N) row-major
    __hip_bfloat16* __restrict__ C,  // (M, N) row-major
    float a_scale,
    float b_scale,
    int M,
    int N,
    int K) {
    const int row_start = blockIdx.y * MFMA_M;
    const int col_start = blockIdx.x * MFMA_N;
    const int lane = threadIdx.x;

    if (row_start >= M || col_start >= N) return;

    // Lane's position in the A/B fragment: low 4 bits select the row (A) or
    // column (B); high 2 bits select which group of 8 contiguous K the lane
    // owns.  Every lane participates — a guard here would zero part of the
    // contraction, which is exactly the bug this rewrite fixed.
    const int lane_rc = lane % MFMA_M;   // A row / B column within the tile
    const int lane_kg = lane / MFMA_M;   // K-group index, 0..3
    const int k_lane = lane_kg * A_FRAG; // this lane's K offset within the tile

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    for (int k_tile = 0; k_tile < K; k_tile += MFMA_K) {
        fp8_t a_frag[A_FRAG] = {};
        fp8_t b_frag[B_FRAG] = {};

        // A fragment: 8 contiguous K at row (row_start + lane_rc).
        const int a_row = row_start + lane_rc;
        if (a_row < M) {
            const int k_base = k_tile + k_lane;
            const int k_avail = (K - k_base) < A_FRAG ? (K - k_base) : A_FRAG;
            for (int e = 0; e < A_FRAG; ++e) {
                a_frag[e] = (e < k_avail) ? A[a_row * K + k_base + e] : (fp8_t)0;
            }
        }

        // B fragment: 8 contiguous K at column (col_start + lane_rc).  B is
        // row-major, so this walks a column with stride N — a strided gather.
        // Correct but slow; an LDS staging pass is what makes it fast, and
        // that is the rewrite this file is deliberately not doing.
        const int b_col = col_start + lane_rc;
        if (b_col < N) {
            const int k_base = k_tile + k_lane;
            for (int e = 0; e < B_FRAG; ++e) {
                const int krow = k_base + e;
                b_frag[e] = (krow < K) ? B[krow * N + b_col] : (fp8_t)0;
            }
        }

        int64_t a_packed, b_packed;
        __builtin_memcpy(&a_packed, a_frag, sizeof(a_packed));
        __builtin_memcpy(&b_packed, b_frag, sizeof(b_packed));

        // MFMA: f32_16x16x32_fp8_fp8.  Consumes 32 K-elements per call.
        //   f32x4 __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
        //       long a, long b, f32x4 c, int cbsz, int abid, int blgp);
        // The control args are 0 for the default ABI.
        acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            a_packed, b_packed, acc, 0, 0, 0);
    }

    // Write back.  C-fragment layout for a 16x16 tile: each lane holds 4
    // floats down the M axis at a single column.
    const int col = col_start + (lane % MFMA_N);
    if (col < N) {
        const int row_base = row_start + (lane / MFMA_N) * C_FRAG;
        const float scale = a_scale * b_scale;
        for (int i = 0; i < C_FRAG; ++i) {
            const int row = row_base + i;
            if (row < M) {
                C[row * N + col] = (__hip_bfloat16)(acc[i] * scale);
            }
        }
    }
}

// Host-callable C entry point.  Declared extern "C" so the pybind layer
// can link it without name mangling games.  The signature uses opaque
// pointer types so the host (.cpp) translation unit can forward-declare
// it without pulling in <hip/...>.
extern "C" void fp8_gemm_launch(
    const uint8_t* dA,  // FP8 storage type is byte-wide
    const uint8_t* dB,
    void* dC,           // __hip_bfloat16*
    float a_scale,
    float b_scale,
    int M,
    int N,
    int K,
    void* stream) {     // hipStream_t
    dim3 block(WAVE);
    dim3 grid((N + MFMA_N - 1) / MFMA_N, (M + MFMA_M - 1) / MFMA_M);
    fp8_gemm_kernel<<<grid, block, 0, reinterpret_cast<hipStream_t>(stream)>>>(
        reinterpret_cast<const fp8_t*>(dA),
        reinterpret_cast<const fp8_t*>(dB),
        reinterpret_cast<__hip_bfloat16*>(dC),
        a_scale, b_scale, M, N, K);
}
