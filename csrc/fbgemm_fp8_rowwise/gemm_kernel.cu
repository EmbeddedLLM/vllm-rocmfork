#include "fp8_gemm_common.cuh"
#include "fp8_gemm_common_splitk.cuh"
#include "kernel/ops.h"

#include <functional>
#include <tuple>
#include <unordered_map>

static constexpr uint32_t DEFAULT_BLOCKS_X = 2;
static constexpr uint32_t DEFAULT_BLOCKS_Y = 2;
static constexpr uint32_t DEFAULT_BLOCKS_Z = 2;
static constexpr uint32_t DEFAULT_MBLOCKS_X = 2;
static constexpr uint32_t DEFAULT_MBLOCKS_Y = 2;
static constexpr uint32_t DEFAULT_CHUNK_K = 256;


at::Tensor f8f8bf16_rowwise_instr1(
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
            TORCH_CHECK(K % (custom_fp8_32x32x16::BLOCK_K * DEFAULT_BLOCKS_Z) == 0, "K must be divisible by 16x");
            LAUNCH_KERNEL_OUTTYPE_32x32x16(_out_dtype, DEFAULT_BLOCKS_X, DEFAULT_BLOCKS_Y, DEFAULT_BLOCKS_Z, DEFAULT_MBLOCKS_X, DEFAULT_MBLOCKS_Y, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor f8f8bf16_rowwise_instr2(
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
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * DEFAULT_BLOCKS_Z) == 0, "K must be divisible by 32x");
            LAUNCH_KERNEL_OUTTYPE_16x16x32(_out_dtype, DEFAULT_BLOCKS_X, DEFAULT_BLOCKS_Y, DEFAULT_BLOCKS_Z, DEFAULT_MBLOCKS_X, DEFAULT_MBLOCKS_Y, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor f8f8bf16_rowwise_instr2_sk(
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
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper_sk(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * DEFAULT_BLOCKS_Z) == 0, "K must be divisible by 32x");
            LAUNCH_KERNEL_OUTTYPE_16x16x32_SK(_out_dtype, DEFAULT_BLOCKS_X, DEFAULT_BLOCKS_Y, DEFAULT_BLOCKS_Z, DEFAULT_MBLOCKS_X, DEFAULT_MBLOCKS_Y, DEFAULT_CHUNK_K, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

#define DISPATCH_32x32x16_WITH_SHAPE(BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y) \
    return custom_fp8_32x32x16::f8f8bf16_rowwise_wrapper( \
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void { \
            TORCH_CHECK(K % (custom_fp8_32x32x16::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 16x"); \
            LAUNCH_KERNEL_OUTTYPE_32x32x16(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
        }, \
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype \
    );

#define DISPATCH_16x16x32_WITH_SHAPE(BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y) \
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper( \
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void { \
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 32x"); \
            LAUNCH_KERNEL_OUTTYPE_16x16x32(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
        }, \
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype \
    );

using KernelType = std::function<
    at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor, std::optional<at::Tensor>, bool, std::optional<at::ScalarType>)>;

struct TripletHasher {
    size_t operator()(const std::tuple<int, int, int> &input) const {
        return (((std::hash<int>()(std::get<0>(input))) ^
                 (std::hash<int>()(std::get<1>(input)) << 1)) >> 1) ^
               (std::hash<int>()(std::get<2>(input)) << 1);
    }
};

static const std::unordered_map<std::tuple<int, int, int>, KernelType, TripletHasher> kernel_dispatch_um = {
    {{  16,  16, 1024}, f8f8bf16_rowwise_16x16x32_11212},
    {{  16,  16, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,  16, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,  16,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,  32, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,  32, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{  16,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,  32,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,  64, 1024}, f8f8bf16_rowwise_16x16x32_11221},
    {{  16,  64, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,  64, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,  64,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16, 128, 1024}, f8f8bf16_rowwise_16x16x32_11221},
    {{  16, 128, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{  16, 128, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16, 128,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16, 256, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16, 256, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16, 256, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16, 256,16384}, f8f8bf16_rowwise_16x16x32_11221},
    {{  16,1024, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,1024, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,1024, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,1024,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,4096, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,4096, 2048}, f8f8bf16_rowwise_16x16x32_11212},
    {{  16,4096, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,4096,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  16,8192, 1024}, f8f8bf16_rowwise_16x16x32_11212},
    {{  16,8192, 2048}, f8f8bf16_rowwise_16x16x32_11212},
    {{  16,8192, 8192}, f8f8bf16_rowwise_16x16x32_11212},
    {{  16,8192,16384}, f8f8bf16_rowwise_16x16x32_11212},
    {{  32,  16, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  16, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{  32,  16, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  16,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  32, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  32, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  32,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  64, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  64, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  64, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,  64,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32, 128, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32, 128, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32, 128, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32, 128,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32, 256, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32, 256, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32, 256, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32, 256,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,1024, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,1024, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,1024, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,1024,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,4096, 1024}, f8f8bf16_rowwise_16x16x32_11212},
    {{  32,4096, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,4096, 8192}, f8f8bf16_rowwise_16x16x32_11221},
    {{  32,4096,16384}, f8f8bf16_rowwise_16x16x32_11212},
    {{  32,8192, 1024}, f8f8bf16_rowwise_16x16x32_11221},
    {{  32,8192, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{  32,8192, 8192}, f8f8bf16_rowwise_16x16x32_11221},
    {{  32,8192,16384}, f8f8bf16_rowwise_16x16x32_11221},
    {{  64,  16, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  16, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  16, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  16,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  32, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  32, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  32,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  64, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  64, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  64, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,  64,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64, 128, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64, 128, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64, 128, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64, 128,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64, 256, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64, 256, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64, 256, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64, 256,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,1024, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,1024, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{  64,1024, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,1024,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,4096, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,4096, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{  64,4096, 8192}, f8f8bf16_rowwise_16x16x32_11221},
    {{  64,4096,16384}, f8f8bf16_rowwise_16x16x32_11221},
    {{  64,8192, 1024}, f8f8bf16_rowwise_16x16x32_11222},
    {{  64,8192, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{  64,8192, 8192}, f8f8bf16_rowwise_16x16x32_11222},
    {{  64,8192,16384}, f8f8bf16_rowwise_16x16x32_11222},
    {{ 128,  16, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  16, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  16, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  16,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  32, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  32, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  32,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  64, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  64, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  64, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,  64,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128, 128, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128, 128, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128, 128, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128, 128,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128, 256, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128, 256, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128, 256, 8192}, f8f8bf16_rowwise_16x16x32_11212},
    {{ 128, 256,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,1024, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,1024, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{ 128,1024, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,1024,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,4096, 1024}, f8f8bf16_rowwise_16x16x32_11221},
    {{ 128,4096, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{ 128,4096, 8192}, f8f8bf16_rowwise_16x16x32_11221},
    {{ 128,4096,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 128,8192, 1024}, f8f8bf16_rowwise_16x16x32_12241},
    {{ 128,8192, 2048}, f8f8bf16_rowwise_16x16x32_12241},
    {{ 128,8192, 8192}, f8f8bf16_rowwise_16x16x32_21214},
    {{ 128,8192,16384}, f8f8bf16_rowwise_16x16x32_12222},
    {{ 256,  16, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  16, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  16, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  16,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  32, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  32, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  32,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  64, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  64, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  64, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,  64,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256, 128, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256, 128, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256, 128, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256, 128,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256, 256, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256, 256, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{ 256, 256, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256, 256,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,1024, 1024}, f8f8bf16_rowwise_16x16x32_11212},
    {{ 256,1024, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{ 256,1024, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,1024,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{ 256,4096, 1024}, f8f8bf16_rowwise_16x16x32_12241},
    {{ 256,4096, 2048}, f8f8bf16_rowwise_16x16x32_12241},
    {{ 256,4096, 8192}, f8f8bf16_rowwise_16x16x32_21214},
    {{ 256,4096,16384}, f8f8bf16_rowwise_16x16x32_21214},
    {{ 256,8192, 1024}, f8f8bf16_rowwise_16x16x32_12141},
    {{ 256,8192, 2048}, f8f8bf16_rowwise_16x16x32_12141},
    {{ 256,8192, 8192}, f8f8bf16_rowwise_16x16x32_22222},
    {{ 256,8192,16384}, f8f8bf16_rowwise_16x16x32_22222},
    {{1024,  16, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  16, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  16, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  16,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  32, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  32, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  32,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  64, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  64, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{1024,  64, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  64,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024, 128, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024, 128, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024, 128, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024, 128,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024, 256, 1024}, f8f8bf16_rowwise_16x16x32_11212},
    {{1024, 256, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{1024, 256, 8192}, f8f8bf16_rowwise_16x16x32_11212},
    {{1024, 256,16384}, f8f8bf16_rowwise_16x16x32_11212},
    {{1024,1024, 1024}, f8f8bf16_rowwise_16x16x32_21214},
    {{1024,1024, 2048}, f8f8bf16_rowwise_16x16x32_21214},
    {{1024,1024, 8192}, f8f8bf16_rowwise_16x16x32_21214},
    {{1024,1024,16384}, f8f8bf16_rowwise_16x16x32_21214},
    {{1024,4096, 1024}, f8f8bf16_rowwise_16x16x32_22142},
    {{1024,4096, 2048}, f8f8bf16_rowwise_16x16x32_22142},
    {{1024,4096, 8192}, f8f8bf16_rowwise_16x16x32_22142},
    {{1024,4096,16384}, f8f8bf16_rowwise_16x16x32_22122},
    {{1024,8192, 1024}, f8f8bf16_rowwise_16x16x32_42124},
    {{1024,8192, 2048}, f8f8bf16_rowwise_16x16x32_22182},
    {{1024,8192, 8192}, f8f8bf16_rowwise_16x16x32_24181},
    {{1024,8192,16384}, f8f8bf16_rowwise_16x16x32_24181},
    {{4096,  16, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{4096,  16, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{4096,  16, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{4096,  16,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{4096,  32, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{4096,  32, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{4096,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{4096,  32,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{4096,  64, 1024}, f8f8bf16_rowwise_16x16x32_11212},
    {{4096,  64, 2048}, f8f8bf16_rowwise_16x16x32_11212},
    {{4096,  64, 8192}, f8f8bf16_rowwise_16x16x32_11212},
    {{4096,  64,16384}, f8f8bf16_rowwise_16x16x32_11212},
    {{4096, 128, 1024}, f8f8bf16_rowwise_16x16x32_11222},
    {{4096, 128, 2048}, f8f8bf16_rowwise_16x16x32_11222},
    {{4096, 128, 8192}, f8f8bf16_rowwise_16x16x32_11222},
    {{4096, 128,16384}, f8f8bf16_rowwise_16x16x32_11222},
    {{4096, 256, 1024}, f8f8bf16_rowwise_16x16x32_12241},
    {{4096, 256, 2048}, f8f8bf16_rowwise_16x16x32_12241},
    {{4096, 256, 8192}, f8f8bf16_rowwise_16x16x32_12221},
    {{4096, 256,16384}, f8f8bf16_rowwise_16x16x32_12221},
    {{4096,1024, 1024}, f8f8bf16_rowwise_16x16x32_22142},
    {{4096,1024, 2048}, f8f8bf16_rowwise_16x16x32_22142},
    {{4096,1024, 8192}, f8f8bf16_rowwise_16x16x32_22142},
    {{4096,1024,16384}, f8f8bf16_rowwise_16x16x32_22142},
    {{4096,4096, 1024}, f8f8bf16_rowwise_16x16x32_24142},
    {{4096,4096, 2048}, f8f8bf16_rowwise_16x16x32_44122},
    {{4096,4096, 8192}, f8f8bf16_rowwise_16x16x32_44122},
    {{4096,4096,16384}, f8f8bf16_rowwise_16x16x32_44122},
    {{4096,8192, 1024}, f8f8bf16_rowwise_16x16x32_24142},
    {{4096,8192, 2048}, f8f8bf16_rowwise_16x16x32_24142},
    {{4096,8192, 8192}, f8f8bf16_rowwise_16x16x32_24142},
    {{4096,8192,16384}, f8f8bf16_rowwise_16x16x32_24181},
    {{8192,  16, 1024}, f8f8bf16_rowwise_16x16x32_11211},
    {{8192,  16, 2048}, f8f8bf16_rowwise_16x16x32_11211},
    {{8192,  16, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{8192,  16,16384}, f8f8bf16_rowwise_16x16x32_11211},
    {{8192,  32, 1024}, f8f8bf16_rowwise_16x16x32_11221},
    {{8192,  32, 2048}, f8f8bf16_rowwise_16x16x32_11221},
    {{8192,  32, 8192}, f8f8bf16_rowwise_16x16x32_11221},
    {{8192,  32,16384}, f8f8bf16_rowwise_16x16x32_11212},
    {{8192,  64, 1024}, f8f8bf16_rowwise_16x16x32_11222},
    {{8192,  64, 2048}, f8f8bf16_rowwise_16x16x32_11222},
    {{8192,  64, 8192}, f8f8bf16_rowwise_16x16x32_11222},
    {{8192,  64,16384}, f8f8bf16_rowwise_16x16x32_11222},
    {{8192, 128, 1024}, f8f8bf16_rowwise_16x16x32_12221},
    {{8192, 128, 2048}, f8f8bf16_rowwise_16x16x32_12221},
    {{8192, 128, 8192}, f8f8bf16_rowwise_16x16x32_12221},
    {{8192, 128,16384}, f8f8bf16_rowwise_16x16x32_21212},
    {{8192, 256, 1024}, f8f8bf16_rowwise_16x16x32_12124},
    {{8192, 256, 2048}, f8f8bf16_rowwise_16x16x32_12142},
    {{8192, 256, 8192}, f8f8bf16_rowwise_16x16x32_12142},
    {{8192, 256,16384}, f8f8bf16_rowwise_16x16x32_22222},
    {{8192,1024, 1024}, f8f8bf16_rowwise_16x16x32_24142},
    {{8192,1024, 2048}, f8f8bf16_rowwise_16x16x32_24142},
    {{8192,1024, 8192}, f8f8bf16_rowwise_16x16x32_24181},
    {{8192,1024,16384}, f8f8bf16_rowwise_16x16x32_24142},
    {{8192,4096, 1024}, f8f8bf16_rowwise_16x16x32_24142},
    {{8192,4096, 2048}, f8f8bf16_rowwise_16x16x32_24142},
    {{8192,4096, 8192}, f8f8bf16_rowwise_16x16x32_24181},
    {{8192,4096,16384}, f8f8bf16_rowwise_16x16x32_24181},
    {{8192,8192, 1024}, f8f8bf16_rowwise_16x16x32_24142},
    {{8192,8192, 2048}, f8f8bf16_rowwise_16x16x32_24142},
    {{8192,8192, 8192}, f8f8bf16_rowwise_16x16x32_24181},
    {{8192,8192,16384}, f8f8bf16_rowwise_16x16x32_44141},
    {{  32,8192, 1280}, f8f8bf16_rowwise_16x16x32_11221},
    {{8192,  32, 1280}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,1024, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{1024,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211},
    {{  32,8192, 7168}, f8f8bf16_rowwise_16x16x32_11212},
    {{8192,  32, 7168}, f8f8bf16_rowwise_16x16x32_11212},
    {{  32,3584, 8192}, f8f8bf16_rowwise_16x16x32_11221},
    {{3584,  32, 8192}, f8f8bf16_rowwise_16x16x32_11211}
};

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    const int N = WQ.size(0);
    const int K = WQ.size(1);

    auto it = kernel_dispatch_um.find({M, N, K});
    if (it != kernel_dispatch_um.end()) {
        return (it->second)(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
    }

    if (M <= 16) {
        if (N > 4096) {
            return f8f8bf16_rowwise_16x16x32_11212(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (N > 1024) {
            if (K <= 2048) {
                return f8f8bf16_rowwise_16x16x32_11212(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (N > 128) {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (N > 64) {
            if (K > 2048) {
                return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_11221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (N > 32) {
            if (K > 1024) {
                return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_11221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        }
    } else if (M <= 32) {
        if (N > 4096) {
            return f8f8bf16_rowwise_16x16x32_11221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (N > 1024) {
            if (K < 8192) {
                return f8f8bf16_rowwise_16x16x32_11221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_11212(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        }
    } else if (M <= 64) {
        if (N >= 8192) {
            return f8f8bf16_rowwise_16x16x32_11222(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (N > 1024) {
            return f8f8bf16_rowwise_16x16x32_11221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (N > 256) {
            if (K > 2048) {
                return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_11212(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        }
    } else if (M <= 128) {
        // Finer tuning between N[4096, 8192]
        if (N >= 8192) {
            if (K > 8192) {
                return f8f8bf16_rowwise_16x16x32_21214(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_12241(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (N >= 4096) {
            return f8f8bf16_rowwise_16x16x32_11221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        }
    } else if (M <= 256) {
        // Finer tuning between N[1024, 8192]
        if (N >= 8192) {
            if (K > 2048) {
                return f8f8bf16_rowwise_16x16x32_22222(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_12141(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (N >= 4096) {
            // or all the way to N > 1024?
            if (K > 2048) {
                return f8f8bf16_rowwise_16x16x32_21214(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_12241(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        }
    } else if (M <= 1024) {
        // Finer tuning between N[1024, 8192]
        if (N >= 8192) {
            if (K > 2048) {
                return f8f8bf16_rowwise_16x16x32_24181(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_42124(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (N >= 4096) {
            if (K > 8192) {
                return f8f8bf16_rowwise_16x16x32_22122(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype); 
            } else {
                return f8f8bf16_rowwise_16x16x32_22142(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype); 
            }
        } else if (N >= 1024) {
            return f8f8bf16_rowwise_16x16x32_21214(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (N >= 256) {
            return f8f8bf16_rowwise_16x16x32_11212(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        }
    } else if (M <= 4096) {
        if (N >= 8192) {
            if (K >= 16384) {
                return f8f8bf16_rowwise_16x16x32_24181(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_24142(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (N >= 4096) {
            return f8f8bf16_rowwise_16x16x32_44122(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (N >= 1024) {
            return f8f8bf16_rowwise_16x16x32_22142(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (N >= 256) {
            if (K > 2048) {
                return f8f8bf16_rowwise_16x16x32_12221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_12241(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (N >= 64) {
            return f8f8bf16_rowwise_16x16x32_11212(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        }
    } else {
        if (N > 8192) {
            if (K > 8192) {
                return f8f8bf16_rowwise_16x16x32_44141(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else if (K > 2048) {
                return f8f8bf16_rowwise_16x16x32_24181(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_24142(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (N > 1024) {
            if (K > 8192) {
                return f8f8bf16_rowwise_16x16x32_24181(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_24142(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (K > 256) {
            return f8f8bf16_rowwise_16x16x32_24142(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (K > 128) {
            if (K > 8192) {
                return f8f8bf16_rowwise_16x16x32_22222(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype); 
            } else {
                return f8f8bf16_rowwise_16x16x32_12142(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else if (K > 64) {
            return f8f8bf16_rowwise_16x16x32_12221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (K > 32) {
            return f8f8bf16_rowwise_16x16x32_11222(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        } else if (K > 16) {
            if (K > 8192) {
                return f8f8bf16_rowwise_16x16x32_11212(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            } else {
                return f8f8bf16_rowwise_16x16x32_11221(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
            }
        } else {
            return f8f8bf16_rowwise_16x16x32_11211(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out_dtype);
        }
    }
}
