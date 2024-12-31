#include <hip/hip_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

#include "cuda_compat.h"

namespace custom_fp8 {

constexpr uint32_t BLOCK_M = 32;
constexpr uint32_t BLOCK_N = 32;
constexpr uint32_t BLOCK_K = 16;
constexpr uint32_t BLOCKS_X = 2;
constexpr uint32_t BLOCKS_Y = 2;
constexpr uint32_t BLOCKS_Z = 2;
constexpr uint32_t MBLOCKS_X = 2;
constexpr uint32_t MBLOCKS_Y = 2;

constexpr uint32_t TBLOCKS_M = BLOCK_M * BLOCKS_X;
constexpr uint32_t TBLOCKS_N = BLOCK_N * BLOCKS_Y;
constexpr uint32_t TBLOCKS_K = BLOCK_K * BLOCKS_Z;
constexpr uint32_t MBLOCKS_M = TBLOCKS_M * MBLOCKS_X;
constexpr uint32_t MBLOCKS_N = TBLOCKS_N * MBLOCKS_Y;

constexpr uint32_t LAUNCH_WARP_SIZE = 64;

// template <typename T>
// __host__ __device__ inline T ceildiv(T a, T b) { return (a + b - 1) / b; }

#define ceildiv(a, b) (((a) + (b) - 1) / (b))

enum class MatrixType {
    A = 0,
    B = 1,
};

__device__ inline void initialize_smem(uint32_t* smem, uint32_t size) {
    uint32_t available_threads = blockDim.x * blockDim.y * blockDim.z;
    uint32_t num_iters = ceildiv(size, available_threads);
    for (int i = 0; i < num_iters; ++i) {
        uint32_t id = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
        if (id < size) {
            smem[id] = 0x0;
        }
    }
}

template <typename T>
__device__ inline void swap_ptr(T* &a, T* &b) {
    T* tmp = a;
    a = b;
    b = tmp;
}

template <typename T>
__device__ inline void load_fp8_gds_to_lds_warp_32x32_packed4u32(
    uint32_t* const lds, const T * const gds, // indexed at warp level
    uint32_t M, uint32_t K,
    uint32_t M_index_warp, uint32_t K_index_warp
) {
    // At the warp level
    using _int4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
    const T* gds_head_elem = gds + (threadIdx.x / 2) * K;
    const _int4* gds_packed = reinterpret_cast<const _int4*>(gds_head_elem);
    gds_packed += (threadIdx.x % 2);

    bool K_within_bound = (K_index_warp + BLOCK_K * (threadIdx.x % 2 + 1)) <= K;
    bool M_within_bound = (M_index_warp + threadIdx.x / 2) < M;

    _int4* lds_packed = reinterpret_cast<_int4*>(lds);
    // lds_packed += (threadIdx.x / 2) * 2 + (threadIdx.x % 2); // threadIdx.x;
    lds_packed += threadIdx.x;

    __syncthreads();

    if (K_within_bound && M_within_bound) {
        *lds_packed = *gds_packed;
    } else {
        *lds_packed = {0, 0, 0, 0};
    }
}

// For loading only
template <MatrixType TMat>
__device__ inline uint32_t* get_lds_load_warp_head_addr(uint32_t* const lds) {
    constexpr uint32_t lds_warp_addr_leap_u32 = 
        (TMat == MatrixType::A) ? BLOCK_M * TBLOCKS_K / 4 : BLOCK_N * TBLOCKS_K / 4;
    uint32_t* lds_warp_addr_head = lds;
    if constexpr (TMat == MatrixType::A) {
        lds_warp_addr_head += lds_warp_addr_leap_u32 * (threadIdx.z + threadIdx.y * BLOCKS_X);
    } else {
        lds_warp_addr_head += lds_warp_addr_leap_u32 * (threadIdx.z + threadIdx.y * BLOCKS_Y);
    }
    return lds_warp_addr_head;
}

template <typename T, MatrixType TMat>
__device__ inline void load_fp8_gds_to_lds_tb_128x32_packed4u32(
    uint32_t* const lds, const T * const gds, // indexed at global level
    uint32_t M, uint32_t K,
    uint32_t M_index_tb, uint32_t K_index_tb
) {
    // At the threadblock level
    uint32_t* lds_warp_addr_head = get_lds_load_warp_head_addr<TMat>(lds);
    
    const uint32_t M_index_warp_head = M_index_tb + threadIdx.y * ((TMat == MatrixType::A) ? TBLOCKS_M : TBLOCKS_N);
    const uint32_t M_index_warp_gds_load = M_index_warp_head + threadIdx.z * ((TMat == MatrixType::A) ? BLOCK_M : BLOCK_N);

    const T* gds_warp_head = gds + M_index_warp_gds_load * K + K_index_tb;
    
    load_fp8_gds_to_lds_warp_32x32_packed4u32<T>(lds_warp_addr_head, gds_warp_head, M, K, M_index_warp_gds_load, K_index_tb);
}

template <int X = BLOCKS_X, int Y = BLOCKS_Y, int Z = 16>
__device__ inline uint32_t get_acc_index(uint32_t i, uint32_t j, uint32_t k = 0) {
    return i * Y * Z + j * Z + k;
}

// Given the lane and gpr ID, return the offset from the warp-level smem head of the 32-bit content
//   to be loaded onto the gpr. smem addresses are indexed in strides of u32
template <MatrixType MatT>
__device__ inline uint32_t get_smem_element_offset_warp_32x16_u32(
    uint32_t lane, uint32_t gpr_num, uint32_t smem_row_stride
) {
    if constexpr (MatT == MatrixType::A) {
        // A matrix
        uint32_t i = lane % 32;
        uint32_t k = 2 * (lane / 32) + gpr_num;
        return i * smem_row_stride + k;
    } else {
        // B matrix
        uint32_t i = lane % 32;
        uint32_t k = 2 * (lane / 32) + gpr_num;
        return i * smem_row_stride + k;
    }
}

__device__ inline void mfma_f32_32x32x16_fp8_fp8(
    const uint32_t * const A_warp_head, // indexed at warp level
    const uint32_t * const B_warp_head, // indexed at warp level
    float* acc_block,
    uint32_t A_row_stride,
    uint32_t B_row_stride
) {
    using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
    using uint32x2 = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;
    union _reg_load {
        uint32x2 regs_;
        long long_;
    };
    // warp-level workers
    for (uint32_t k_inner_iter = 0; k_inner_iter < BLOCKS_Z; ++k_inner_iter) {
        const uint32_t* A_warp_head_inner = A_warp_head + k_inner_iter * (BLOCK_K / 4);
        const uint32_t* B_warp_head_inner = B_warp_head + k_inner_iter * (BLOCK_K / 4);
        _reg_load a_regs;
        _reg_load b_regs;
        for (int reg = 0; reg < 2; ++reg) {
            a_regs.regs_[reg] = A_warp_head_inner[get_smem_element_offset_warp_32x16_u32<MatrixType::A>(threadIdx.x, reg, A_row_stride)];
            b_regs.regs_[reg] = B_warp_head_inner[get_smem_element_offset_warp_32x16_u32<MatrixType::B>(threadIdx.x, reg, B_row_stride)];
        }
        floatx16* acc_block_f16 = reinterpret_cast<floatx16*>(acc_block);
        *acc_block_f16 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_regs.long_, b_regs.long_, *acc_block_f16, 0, 0, 0);
    }
}

__device__ inline void mfma_f32_64x64x32_fp8_fp8(
    const uint32_t * const A_head_tb, // indexed at threadblock level
    const uint32_t * const B_head_tb, // indexed at threadblock level
    float* acc, // blocksX x blocksY x 16,
    uint32_t A_row_stride, // in u32
    uint32_t B_row_stride  // in u32 (BLOCK_K / 4) * BLOCKS_Z
) {
    uint32_t warp_x = threadIdx.y;
    uint32_t warp_y = threadIdx.z;
    for (uint32_t x_iter = 0; x_iter < BLOCKS_X; ++x_iter) {
        uint32_t x_load_block = warp_x * BLOCKS_X + x_iter;
        const uint32_t* A_warp_head = A_head_tb + x_load_block * BLOCK_M * A_row_stride;
        for (uint32_t y_iter = 0; y_iter < BLOCKS_Y; ++y_iter) {
            uint32_t y_load_block = warp_y * BLOCKS_Y + y_iter;
            const uint32_t* B_warp_head = B_head_tb + y_load_block * BLOCK_N * B_row_stride;
            float* acc_block = acc + get_acc_index(x_iter, y_iter);
            mfma_f32_32x32x16_fp8_fp8(A_warp_head, B_warp_head, acc_block, A_row_stride, B_row_stride);
        }
    }
}

template <typename TF32, MatrixType MatT>
__device__ inline void load_scale_gds_to_lds_128(
    const TF32 * const gds, // indexed at global level
    float* lds,
    uint32_t tb_head_index,
    uint32_t size
) {
    const TF32* gds_head_tb = gds + tb_head_index;
    bool active = false;
    uint32_t offset = 0;
    uint32_t num_iters = 1;
    if constexpr (MatT == MatrixType::A) {
        offset = threadIdx.y * TBLOCKS_M + threadIdx.x;
        num_iters = TBLOCKS_M / LAUNCH_WARP_SIZE;
        active = !(threadIdx.z % BLOCKS_Y) && (offset + num_iters - 1 < size);
    } else {
        offset = threadIdx.z * TBLOCKS_N + threadIdx.x;
        num_iters = TBLOCKS_N / LAUNCH_WARP_SIZE;
        active = !(threadIdx.y % BLOCKS_X) && (offset + num_iters - 1 < size);
    }

    __syncthreads();
    if (active) {
#pragma unroll
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            uint32_t offset_iter = offset + iter;
            lds[offset_iter] = (offset_iter < size) ? static_cast<float>(gds_head_tb[offset + iter]) : 0.0f;
        }
    }
    __syncthreads();
}

template <typename TF32>
__device__ inline void apply_scale(
    const float * const lds_xscale,
    const TF32 * const gds_wscale, // Indexed at global level
    float* acc,
    uint32_t wscale_tb_head_index,
    uint32_t wscale_size
) {
    // Each warp applies scales to a 64x64 block
    const uint32_t lane_id = threadIdx.x;
    uint32_t gds_wscale_warp_offset = wscale_tb_head_index + threadIdx.z * BLOCKS_Y * BLOCK_N;
    const float* lds_xscale_warp = lds_xscale + threadIdx.y * BLOCKS_X * BLOCK_M;
#pragma unroll
    for (uint32_t warp_y = 0; warp_y < BLOCKS_Y; ++warp_y) {
        uint32_t gds_wscale_warp_iter_offset = gds_wscale_warp_offset + warp_y * BLOCK_N;
        uint32_t wscale_offset_thread = gds_wscale_warp_iter_offset + (lane_id % 32);
        __syncthreads();
        float wscale = (wscale_offset_thread < wscale_size) ? 
            static_cast<float>(gds_wscale[wscale_offset_thread]) : 0.0f;

        for (uint32_t warp_x = 0; warp_x < BLOCKS_X; ++warp_x) {
            float* acc_warp = acc + get_acc_index(warp_x, warp_y);
            const float* lds_xscale_warp_iter = lds_xscale_warp + warp_x * BLOCK_M;

#pragma unroll
            for (uint32_t reg = 0; reg < 16; ++reg) {
                acc_warp[reg] *= wscale;
                acc_warp[reg] *= lds_xscale_warp_iter[(8 * (reg / 4) % 32) + 4 * (lane_id / 32) + (reg % 4)];
            }
        }
    }
}

template <typename TY>
__device__ inline void store_acc_to_gds_transposed(
    TY* y_gds,
    float* acc,
    uint32_t M_head_index, // col head id
    uint32_t N_head_index, // row head id
    uint32_t M,
    uint32_t N,
    uint32_t y_row_stride // should be M
) {
    // constexpr uint32_t pack_factor = 4; // Must be 4
    const uint32_t lane_id = threadIdx.x;
    const uint32_t M_warp_head_offset = M_head_index + threadIdx.y * TBLOCKS_M;
    const uint32_t N_warp_head_offset = N_head_index + threadIdx.z * TBLOCKS_N;
    for (uint32_t warp_m = 0; warp_m < BLOCKS_X; ++warp_m) {
        uint32_t M_warp_iter_head_offset = M_warp_head_offset + warp_m * BLOCK_M;
        for (uint32_t warp_n = 0; warp_n < BLOCKS_Y; ++warp_n) {
            uint32_t N_warp_iter_head_offset = N_warp_head_offset + warp_n * BLOCK_N;
            uint32_t N_lane_reg_offset = N_warp_iter_head_offset + (lane_id % 32);
            if (N_lane_reg_offset >= N) { continue; }
            uint32_t N_offset_strided = N_lane_reg_offset * y_row_stride;

            for (uint32_t reg = 0; reg < 16; reg += 4) {
                uint32_t M_lane_reg_offset = M_warp_iter_head_offset + (8 * (reg / 4) % 32) + 4 * (lane_id / 32) + (reg % 4);
                if (M_lane_reg_offset >= M) { continue; }
                if constexpr (sizeof(TY) == 2) {
                    using TY4 = __attribute__((__vector_size__(4 * 2))) uint16_t;
                    TY buffer[4];
                    uint32_t acc_index_base = get_acc_index(warp_m, warp_n);
                    for (int rr = 0; rr < 4; ++rr) {
                        buffer[rr] = static_cast<TY>(acc[get_acc_index(warp_m, warp_n) + reg + rr]);
                    }
                    *(reinterpret_cast<TY4 *>(y_gds + N_offset_strided + M_lane_reg_offset)) = *(reinterpret_cast<TY4 *>(buffer));
                } else {
                    *(reinterpret_cast<int4 *>(y_gds + N_offset_strided + M_lane_reg_offset)) = 
                        *(reinterpret_cast<int4 *>(acc + get_acc_index(warp_m, warp_n, reg)));
                }
            }

        }
    }
}

template <typename TF8, typename TF32, typename TY>
__global__ void f8f8f16_rowwise_kernel(
    const TF8 * const xq,
    const TF8 * const wq,
    const TF32 * const x_scale,
    const TF32 * const w_scale,
    TY* y,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    constexpr uint32_t A_block_size_u32 = BLOCK_M * BLOCK_K / (4 / sizeof(TF8));
    constexpr uint32_t A_warp_block_size_u32 = A_block_size_u32 * BLOCKS_X * BLOCKS_Z;
    constexpr uint32_t A_tile_block_size_u32 = A_warp_block_size_u32 * MBLOCKS_X;
    constexpr uint32_t B_block_size_u32 = BLOCK_N * BLOCK_K / (4 / sizeof(TF8));
    constexpr uint32_t B_warp_block_size_u32 = B_block_size_u32 * BLOCKS_Y * BLOCKS_Z;
    constexpr uint32_t B_tile_block_size_u32 = B_warp_block_size_u32 * MBLOCKS_Y;

    __shared__ uint32_t A_shared[A_tile_block_size_u32 * 2];
    __shared__ uint32_t B_shared[B_tile_block_size_u32 * 2]; // transposed

    constexpr uint32_t A_row_stride = BLOCK_K / (4 / sizeof(TF8)) * BLOCKS_Z;
    constexpr uint32_t B_row_stride = BLOCK_K / (4 / sizeof(TF8)) * BLOCKS_Z;

    uint32_t* A_shared_load = A_shared;
    uint32_t* A_shared_eval = A_shared + A_tile_block_size_u32;
    uint32_t* B_shared_load = B_shared;
    uint32_t* B_shared_eval = B_shared + A_tile_block_size_u32;

    initialize_smem(A_shared_load, A_tile_block_size_u32);
    initialize_smem(B_shared_load, B_tile_block_size_u32);

    float acc[BLOCKS_X * BLOCKS_Y * 16];
    for (uint32_t i = 0; i < BLOCKS_X * BLOCKS_Y * 16; ++i) {
        acc[i] = 0.0f;
    }

    const uint32_t M_index_tile = blockIdx.x * MBLOCKS_M; // head of threadblock
    const uint32_t N_index_tile = blockIdx.y * MBLOCKS_N; // head of threadblock

    const uint32_t k_iters = ceildiv(K, BLOCK_K * BLOCKS_Z);
    uint32_t K_index_tile = 0;

    __syncthreads();

    // Iteration #0 loading
    load_fp8_gds_to_lds_tb_128x32_packed4u32<TF8, MatrixType::A>(A_shared_load, xq, M, K, M_index_tile, K_index_tile);
    load_fp8_gds_to_lds_tb_128x32_packed4u32<TF8, MatrixType::B>(B_shared_load, wq, N, K, N_index_tile, K_index_tile);

    swap_ptr(A_shared_load, A_shared_eval);
    swap_ptr(B_shared_load, B_shared_eval);

    __syncthreads();

    for (int kk = 1; kk < k_iters; ++kk) {
        // load
        K_index_tile += BLOCK_K * BLOCKS_Z;
        load_fp8_gds_to_lds_tb_128x32_packed4u32<TF8, MatrixType::A>(A_shared_load, xq, M, K, M_index_tile, K_index_tile);
        load_fp8_gds_to_lds_tb_128x32_packed4u32<TF8, MatrixType::B>(B_shared_load, wq, N, K, N_index_tile, K_index_tile);

        // compute mm
        mfma_f32_64x64x32_fp8_fp8(A_shared_eval, B_shared_eval, acc, A_row_stride, B_row_stride);

        // swap
        swap_ptr(A_shared_load, A_shared_eval);
        swap_ptr(B_shared_load, B_shared_eval);

        __syncthreads();
    }
    // Iteration #-1 computing
    mfma_f32_64x64x32_fp8_fp8(A_shared_eval, B_shared_eval, acc, A_row_stride, B_row_stride);

    // Apply scales
    float* x_scale_shared = reinterpret_cast<float*>(A_shared_load);
    load_scale_gds_to_lds_128<TF32, MatrixType::A>(x_scale, x_scale_shared, M_index_tile, M);
    apply_scale(x_scale_shared, w_scale, acc, N_index_tile, N);
    __syncthreads();

    // Save the result to gds in transpose to facilitate coalescence
    store_acc_to_gds_transposed(y, acc, M_index_tile, N_index_tile, M, N, M);
}

at::Tensor f8f8bf16_rowwise_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y
) {
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);

    dim3 grid(ceildiv(M, MBLOCKS_M), ceildiv(N, MBLOCKS_N), 1);
    dim3 block(LAUNCH_WARP_SIZE, MBLOCKS_X, MBLOCKS_Y);

    auto stream{torch::hip::getCurrentHIPStream().stream()};

#define LAUNCH_KERNEL(TFY) \
        { \
        auto kernel = f8f8f16_rowwise_kernel<uint8_t, float, TFY>; \
        kernel<<<grid, block, 0, stream>>>( \
            reinterpret_cast<uint8_t*>(XQ.data_ptr()), \
            reinterpret_cast<uint8_t*>(WQ.data_ptr()), \
            reinterpret_cast<float*>(x_scale.data_ptr()), \
            reinterpret_cast<float*>(w_scale.data_ptr()), \
            reinterpret_cast<TFY*>(Y.data_ptr()), \
            M, N, K \
        ); \
        }

    if (Y.dtype() == at::kFloat) {
        LAUNCH_KERNEL(float)
    } else if (Y.dtype() == at::kHalf) {
        LAUNCH_KERNEL(__half)
    } else if (Y.dtype() == at::kBFloat16) {
        LAUNCH_KERNEL(__hip_bfloat16)
    } else {
        AT_ERROR("Not implemented output datatype. Must be one of {float, half, bfloat16}.");
    }
    
    return Y;
}

at::Tensor f8f8bf16_rowwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // not calculated
    bool use_fast_accum,
    std::optional<at::Tensor> output = std::nullopt,
    std::optional<at::ScalarType> out_dtype = std::nullopt) {
    // Check that input datatypes are valid.
    TORCH_CHECK(
        (XQ.dtype() == at::kFloat8_e4m3fnuz) &&
            (WQ.dtype() == at::kFloat8_e4m3fnuz),
        "Inputs must be type float8_e4m3fnuz.");
    TORCH_CHECK(
        (x_scale.dtype() == at::kFloat) && (w_scale.dtype() == at::kFloat),
        "Scales must be float32.");
    TORCH_CHECK(use_fast_accum, "AMD does not support disabling use_fast_accum.");

    // Check inputs are in expected format.
    TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
    TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

    // XQ: M x K
    // WQ: N x K
    // output: M x N
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);

    TORCH_CHECK((K % 16) == 0, 
        "Cases where K is not divisible by 16 has not been implemented.");

    at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;

    // Prepare output tensor if needed.
    at::Tensor Y;
    if (output.has_value()) {
        Y = output.value();
        // Make sure the provided output has the proper shape and dtype.
        if (Y.dim() >= 3) {
            int B = size_to_dim_(Y.dim() - 2, Y.sizes());
            int Y_M = Y.size(Y.dim() - 1);
            int Y_N = Y.size(Y.dim() - 2);
            TORCH_CHECK(Y_M*B == M && Y_N == N, "Y must be transposed");
        } else if (Y.dim() == 2) {
            int Y_M = Y.size(Y.dim() - 1);
            int Y_N = Y.size(Y.dim() - 2);
            TORCH_CHECK(Y_M == M && Y_N == N, "Y must be transposed");
        } else {
            AT_ERROR("Output should at least have two dimensions");
        }
        TORCH_CHECK(Y.dtype() == _out_dtype);
    } else {
        // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
        // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
        if (XQ.dim() >= 3) {
            int B = size_to_dim_(XQ.dim() - 2, XQ.sizes());
            int X_M = XQ.size(XQ.dim() - 2);
            int W_N = WQ.size(WQ.dim() - 1);
            Y = at::empty({B, W_N, X_M}, XQ.options().dtype(_out_dtype));
        } else if (XQ.dim() == 2) {
            int X_M = XQ.size(XQ.dim() - 2);
            int W_N = WQ.size(WQ.dim() - 2);
            Y = at::empty({W_N, X_M}, XQ.options().dtype(_out_dtype));
        } else {
            AT_ERROR("Output should at least have two dimensions");
        }
    }

    return f8f8bf16_rowwise_impl(XQ, WQ, x_scale, w_scale, Y);
}

} // namespace custom_fp8


at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype
) {
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8::f8f8bf16_rowwise_wrapper(
        XQ, WQ, x_scale, w_scale, bias, use_fast_accum, std::nullopt, out_dtype);
}
