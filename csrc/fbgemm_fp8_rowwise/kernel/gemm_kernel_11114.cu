
#include <cuda_runtime.h>
#include "../fp8_gemm_common_hip.cuh"

constexpr uint32_t BLOCKS_X = 1;
constexpr uint32_t BLOCKS_Y = 1;
constexpr uint32_t BLOCKS_Z = 1;
constexpr uint32_t MBLOCKS_X = 1;
constexpr uint32_t MBLOCKS_Y = 4;

at::Tensor f8f8bf16_rowwise_32x32x16_11114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_32x32x16::f8f8bf16_rowwise_wrapper(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_32x32x16::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 16x");
            LAUNCH_KERNEL_OUTTYPE_32x32x16(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor f8f8bf16_rowwise_16x16x32_11114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 32x");
            LAUNCH_KERNEL_OUTTYPE_16x16x32(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}
