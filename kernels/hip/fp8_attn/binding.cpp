// pybind11 / torch extension surface for the gfx942 FP8 GEMM kernel.
//
// This is the host-side translation unit.  It deliberately avoids HIP
// headers (which trip g++ in PyTorch's hipify-on-the-fly build path) and
// forward-declares only the C ABI launch function exported from
// fp8_gemm.hip.
//
// Real perf path: kernels/triton_kernels/fp8_flash_attn.py.

#include <torch/extension.h>
#include <cstdint>

// Opaque C ABI: matches the extern "C" declaration in fp8_gemm.hip.
extern "C" void fp8_gemm_launch(
    const uint8_t* dA,
    const uint8_t* dB,
    void* dC,  // __hip_bfloat16*
    float a_scale,
    float b_scale,
    int M,
    int N,
    int K,
    void* stream);  // hipStream_t

torch::Tensor fp8_gemm(
    torch::Tensor A,
    torch::Tensor B,
    double a_scale,
    double b_scale) {
    TORCH_CHECK(A.dim() == 2, "A must be 2-D");
    TORCH_CHECK(B.dim() == 2, "B must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0), "K mismatch");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat8_e4m3fnuz, "A must be float8_e4m3fnuz");
    TORCH_CHECK(B.scalar_type() == torch::kFloat8_e4m3fnuz, "B must be float8_e4m3fnuz");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(A.device());
    torch::Tensor C = torch::empty({M, N}, opts);

    // Pass the default stream (null) — sync stream selection is a future
    // optimization.  The current PyTorch HIP stream API name varies between
    // 2.x releases; passing nullptr lets the kernel use the device default.
    fp8_gemm_launch(
        reinterpret_cast<const uint8_t*>(A.data_ptr()),
        reinterpret_cast<const uint8_t*>(B.data_ptr()),
        C.data_ptr(),
        static_cast<float>(a_scale),
        static_cast<float>(b_scale),
        M, N, K, nullptr);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_gemm", &fp8_gemm,
          "FP8 GEMM (proof-of-life HIP kernel for gfx942 MFMA)");
}
