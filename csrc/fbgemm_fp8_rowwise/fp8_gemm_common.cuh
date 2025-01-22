#pragma once

#include <hip/hip_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

#include "cuda_compat.h"

#if defined(__HIPCC__) && \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
  #define __HIP__MI300__
#endif

constexpr uint32_t LAUNCH_WARP_SIZE = 64;

#define ceildiv(a, b) (((a) + (b) - 1) / (b))

__device__ inline float convert_fp8_to_float(const uint8_t* in) {
    const __hip_fp8_storage_t* in_cvt = reinterpret_cast<const __hip_fp8_storage_t*>(in);
    __half hf = __hip_cvt_fp8_to_halfraw(*in_cvt, __HIP_E4M3_FNUZ);
    return __half2float(hf);
}

enum class MatrixType {
    A = 0,
    B = 1,
};

__device__ inline void initialize_smem(uint32_t* smem, uint32_t size) {
    uint32_t available_threads = blockDim.x * blockDim.y * blockDim.z;
    uint32_t num_iters = ceildiv(size, available_threads);
    for (int i = 0; i < num_iters; ++i) {
        uint32_t id = i * available_threads + threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
        if (id < size) {
            smem[id] = 0x0;
        }
    }
}

// K must be divisible by 16
template <
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int BLOCKS_X, int BLOCKS_Y, int BLOCKS_Z,
    int MBLOCKS_X, int MBLOCKS_Y,
    typename TF8, MatrixType TMat
>
__device__ inline void load_fp8_gds_to_lds_packed4u32(
    uint32_t * const lds, const TF8 * const gds, // indexed at global level
    uint32_t M, uint32_t K,
    uint32_t M_index_tb, uint32_t K_index_tb
) {
    // At threadblock level
    constexpr uint32_t num_elems_per_row = BLOCK_K * BLOCKS_Z / (16 / sizeof(TF8));
    constexpr uint32_t available_workers = LAUNCH_WARP_SIZE * MBLOCKS_X * MBLOCKS_Y;
    constexpr uint32_t num_rows_per_iter = ceildiv(available_workers, num_elems_per_row);
    constexpr uint32_t num_rows = (TMat == MatrixType::A) ? BLOCK_M * BLOCKS_X * MBLOCKS_X : BLOCK_N * BLOCKS_Y * MBLOCKS_Y;
    constexpr uint32_t num_iters = (num_rows > num_rows_per_iter) ? ceildiv(num_rows, num_rows_per_iter) : 1;

    using uint32x4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
    uint32x4* lds_packed = reinterpret_cast<uint32x4*>(lds);

    const TF8* gds_head_tb = gds + M_index_tb * K + K_index_tb;

    const uint32_t thread_id_flattened = threadIdx.x + LAUNCH_WARP_SIZE * (threadIdx.y + MBLOCKS_X * threadIdx.z);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t row_id = iter * num_rows_per_iter + thread_id_flattened / num_elems_per_row;
        const uint32_t col_id = thread_id_flattened % num_elems_per_row;
        const bool row_within_bound_tb = row_id < num_rows;
        
        const uint32x4* gds_elem = reinterpret_cast<const uint32x4*>(gds_head_tb + row_id * K) + col_id;
        
        if (row_within_bound_tb) { 
            if (row_id < (M - M_index_tb)) {
                *(lds_packed + row_id * num_elems_per_row + col_id) = *gds_elem;
            } else {
                *(lds_packed + row_id * num_elems_per_row + col_id) = {0, 0, 0, 0};
            }
        }
    }
}

template <typename T>
__device__ inline void swap_ptr(T* &a, T* &b) {
    T* tmp = a;
    a = b;
    b = tmp;
}

template <
    int BLOCK_M, int BLOCK_N,
    int BLOCKS_X, int BLOCKS_Y,
    int MBLOCKS_X, int MBLOCKS_Y,
    typename TF32, MatrixType MatT
>
__device__ inline void load_scale_gds_to_lds_vanilla(
    const TF32 * const gds, // indexed at global level
    float* lds,
    uint32_t tb_head_index,
    uint32_t size
) {
    constexpr uint32_t num_elems = (MatT == MatrixType::A) ? BLOCK_M * BLOCKS_X * MBLOCKS_X : BLOCK_N * BLOCKS_Y * MBLOCKS_Y;
    constexpr uint32_t available_workers = LAUNCH_WARP_SIZE * MBLOCKS_X * MBLOCKS_Y;
    constexpr uint32_t num_iters = ceildiv(num_elems, available_workers);

    const uint32_t thread_id_flattened = threadIdx.x + LAUNCH_WARP_SIZE * (threadIdx.y + MBLOCKS_X * threadIdx.z);

#pragma unroll
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t item_id = thread_id_flattened + iter * available_workers;
        const uint32_t global_item_id = tb_head_index + item_id;
        if (item_id < num_elems && global_item_id < size) {
            lds[item_id] = static_cast<float>(gds[global_item_id]);
        }
    }
}

template <typename TY, int NUM_ITEMS>
__device__ inline void store_regs4_to_gds(
    TY* gds_head, // indexed at the lane-reg head
    const float* acc_index_base_reg // indexed at the lane-reg head
) {
    if constexpr (sizeof(TY) == 2) {
        using TYO = __attribute__((__vector_size__(NUM_ITEMS * 2))) uint16_t;
        TY buffer[NUM_ITEMS];
#pragma unroll
        for (int rr = 0; rr < NUM_ITEMS; ++rr) {
            buffer[rr] = static_cast<TY>(acc_index_base_reg[rr]);
        }
        if constexpr (NUM_ITEMS == 3) {
            using TY2 = __attribute__((__vector_size__(2 * 2))) uint16_t;
            *(reinterpret_cast<TY2 *>(gds_head)) = *(reinterpret_cast<TY2 *>(buffer));
            *(reinterpret_cast<TY *>(gds_head) + 2) = buffer[2];
        } else {
            *(reinterpret_cast<TYO *>(gds_head)) = *(reinterpret_cast<TYO *>(buffer));
        }
    } else {
        using TYO = __attribute__((__vector_size__(NUM_ITEMS * 4))) uint32_t;
        if constexpr (NUM_ITEMS == 3) {
            // Handling the case where copying 12 words introduces padding of zeros to 16 words
            using TY2 = __attribute__((__vector_size__(2 * 4))) uint32_t;
            *(reinterpret_cast<TY2 *>(gds_head)) = 
                            *(reinterpret_cast<const TY2 *>(acc_index_base_reg));
            *(reinterpret_cast<TY *>(gds_head) + 2) = *(reinterpret_cast<const TY *>(acc_index_base_reg + 2));
        } else {
            *(reinterpret_cast<TYO *>(gds_head)) = 
                            *(reinterpret_cast<const TYO *>(acc_index_base_reg));
        }
    }
}

namespace custom_fp8_32x32x16 {

constexpr uint32_t BLOCK_M = 32;
constexpr uint32_t BLOCK_N = 32;
constexpr uint32_t BLOCK_K = 16;

template <int BLOCKS_X, int BLOCKS_Y, int Z = 16>
__device__ inline uint32_t get_acc_index(uint32_t i, uint32_t j, uint32_t k = 0) {
    return (i * BLOCKS_Y + j) * Z + k;
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

template <int BLOCKS_Z>
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
#pragma unroll
    for (uint32_t k_inner_iter = 0; k_inner_iter < BLOCKS_Z; ++k_inner_iter) {
        const uint32_t* A_warp_head_inner = A_warp_head + k_inner_iter * (BLOCK_K / 4);
        const uint32_t* B_warp_head_inner = B_warp_head + k_inner_iter * (BLOCK_K / 4);
        _reg_load a_regs;
        _reg_load b_regs;
#pragma unroll
        for (int reg = 0; reg < 2; ++reg) {
            a_regs.regs_[reg] = A_warp_head_inner[get_smem_element_offset_warp_32x16_u32<MatrixType::A>(threadIdx.x, reg, A_row_stride)];
            b_regs.regs_[reg] = B_warp_head_inner[get_smem_element_offset_warp_32x16_u32<MatrixType::B>(threadIdx.x, reg, B_row_stride)];
        }
        floatx16* acc_block_f16 = reinterpret_cast<floatx16*>(acc_block);
        *acc_block_f16 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_regs.long_, b_regs.long_, *acc_block_f16, 0, 0, 0);
    }
}

template <int BLOCKS_X, int BLOCKS_Y, int BLOCKS_Z>
__device__ inline void mfma_f32_64x64x32_fp8_fp8(
    const uint32_t * const A_head_tb, // indexed at threadblock level
    const uint32_t * const B_head_tb, // indexed at threadblock level
    float* acc, // blocksX x blocksY x 16,
    uint32_t A_row_stride, // in u32
    uint32_t B_row_stride  // in u32 (BLOCK_K / 4) * BLOCKS_Z
) {
    uint32_t warp_x = threadIdx.y;
    uint32_t warp_y = threadIdx.z;
#pragma unroll
    for (uint32_t x_iter = 0; x_iter < BLOCKS_X; ++x_iter) {
        uint32_t x_load_block = warp_x * BLOCKS_X + x_iter;
        const uint32_t* A_warp_head = A_head_tb + x_load_block * BLOCK_M * A_row_stride;
#pragma unroll
        for (uint32_t y_iter = 0; y_iter < BLOCKS_Y; ++y_iter) {
            uint32_t y_load_block = warp_y * BLOCKS_Y + y_iter;
            const uint32_t* B_warp_head = B_head_tb + y_load_block * BLOCK_N * B_row_stride;
            float* acc_block = acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 16>(x_iter, y_iter);
            mfma_f32_32x32x16_fp8_fp8<BLOCKS_Z>(A_warp_head, B_warp_head, acc_block, A_row_stride, B_row_stride);
        }
    }
}

template <int BLOCKS_X, int BLOCKS_Y, int BLOCKS_Z, typename TF32>
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
#pragma unroll
        for (uint32_t warp_x = 0; warp_x < BLOCKS_X; ++warp_x) {
            float* acc_warp = acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 16>(warp_x, warp_y);
            const float* lds_xscale_warp_iter = lds_xscale_warp + warp_x * BLOCK_M;

#pragma unroll
            for (uint32_t reg = 0; reg < 16; ++reg) {
                acc_warp[reg] *= wscale;
                acc_warp[reg] *= lds_xscale_warp_iter[(8 * (reg / 4) % 32) + 4 * (lane_id / 32) + (reg % 4)];
            }
        }
    }
}

template <int BLOCKS_X, int BLOCKS_Y, typename TY>
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
    const uint32_t M_warp_head_offset = M_head_index + threadIdx.y * BLOCK_M * BLOCKS_X;
    const uint32_t N_warp_head_offset = N_head_index + threadIdx.z * BLOCK_N * BLOCKS_Y;
#pragma unroll
    for (uint32_t warp_m = 0; warp_m < BLOCKS_X; ++warp_m) {
        uint32_t M_warp_iter_head_offset = M_warp_head_offset + warp_m * BLOCK_M;
#pragma unroll
        for (uint32_t warp_n = 0; warp_n < BLOCKS_Y; ++warp_n) {
            uint32_t N_warp_iter_head_offset = N_warp_head_offset + warp_n * BLOCK_N;
            uint32_t N_lane_reg_offset = N_warp_iter_head_offset + (lane_id % 32);
            if (N_lane_reg_offset >= N) { continue; }
            uint32_t N_offset_strided = N_lane_reg_offset * y_row_stride;
#pragma unroll
            for (uint32_t reg = 0; reg < 16; reg += 4) {
                uint32_t M_lane_reg_offset = M_warp_iter_head_offset + (8 * (reg / 4) % 32) + 4 * (lane_id / 32) + (reg % 4);
                if (M_lane_reg_offset >= M) { continue; }
                const uint32_t num_items = M - M_lane_reg_offset;
                if (num_items > 3) {
                    store_regs4_to_gds<TY, 4>(y_gds + N_offset_strided + M_lane_reg_offset, acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 16>(warp_m, warp_n) + reg);
                } else if (num_items == 3) {
                    store_regs4_to_gds<TY, 3>(y_gds + N_offset_strided + M_lane_reg_offset, acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 16>(warp_m, warp_n) + reg);
                } else if (num_items == 2) {
                    store_regs4_to_gds<TY, 2>(y_gds + N_offset_strided + M_lane_reg_offset, acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 16>(warp_m, warp_n) + reg);
                } else {
                    store_regs4_to_gds<TY, 1>(y_gds + N_offset_strided + M_lane_reg_offset, acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 16>(warp_m, warp_n) + reg);
                }
            }

        }
    }
}

template <
    int BLOCKS_X, int BLOCKS_Y, int BLOCKS_Z,
    int MBLOCKS_X, int MBLOCKS_Y,
    typename TF8, typename TF32, typename TY
>
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
    constexpr uint32_t TBLOCKS_M = BLOCK_M * BLOCKS_X;
    constexpr uint32_t TBLOCKS_N = BLOCK_N * BLOCKS_Y;
    constexpr uint32_t TBLOCKS_K = BLOCK_K * BLOCKS_Z;
    constexpr uint32_t MBLOCKS_M = TBLOCKS_M * MBLOCKS_X;
    constexpr uint32_t MBLOCKS_N = TBLOCKS_N * MBLOCKS_Y;

    constexpr uint32_t A_block_size_u32 = BLOCK_M * BLOCK_K / (4 / sizeof(TF8));
    constexpr uint32_t A_warp_block_size_u32 = A_block_size_u32 * BLOCKS_X * BLOCKS_Z;
    constexpr uint32_t A_tile_block_size_u32 = A_warp_block_size_u32 * MBLOCKS_X;
    constexpr uint32_t B_block_size_u32 = BLOCK_N * BLOCK_K / (4 / sizeof(TF8));
    constexpr uint32_t B_warp_block_size_u32 = B_block_size_u32 * BLOCKS_Y * BLOCKS_Z;
    constexpr uint32_t B_tile_block_size_u32 = B_warp_block_size_u32 * MBLOCKS_Y;

    __shared__ __attribute__((aligned(512))) uint32_t A_shared[A_tile_block_size_u32 * 2];
    __shared__ __attribute__((aligned(512))) uint32_t B_shared[B_tile_block_size_u32 * 2]; // transposed

    constexpr uint32_t A_row_stride = BLOCK_K / (4 / sizeof(TF8)) * BLOCKS_Z;
    constexpr uint32_t B_row_stride = BLOCK_K / (4 / sizeof(TF8)) * BLOCKS_Z;

    uint32_t* A_shared_load = A_shared;
    uint32_t* A_shared_eval = A_shared + A_tile_block_size_u32;
    uint32_t* B_shared_load = B_shared;
    uint32_t* B_shared_eval = B_shared + B_tile_block_size_u32;

    initialize_smem(A_shared_load, A_tile_block_size_u32);
    initialize_smem(B_shared_load, B_tile_block_size_u32);

    float acc[BLOCKS_X * BLOCKS_Y * 16];
#pragma unroll
    for (uint32_t i = 0; i < BLOCKS_X * BLOCKS_Y * 16; ++i) {
        acc[i] = 0.0f;
    }

    const uint32_t M_index_tile = blockIdx.x * MBLOCKS_M; // head of threadblock
    const uint32_t N_index_tile = blockIdx.y * MBLOCKS_N; // head of threadblock

    const uint32_t k_iters = ceildiv(K, BLOCK_K * BLOCKS_Z);

    __syncthreads();

    // Iteration #0 loading
    load_fp8_gds_to_lds_packed4u32<BLOCK_M, BLOCK_N, BLOCK_K, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, TF8, MatrixType::A>(
        A_shared_load, xq, M, K, M_index_tile, 0
    );
    load_fp8_gds_to_lds_packed4u32<BLOCK_M, BLOCK_N, BLOCK_K, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, TF8, MatrixType::B>(
        B_shared_load, wq, N, K, N_index_tile, 0
    );

    swap_ptr(A_shared_load, A_shared_eval);
    swap_ptr(B_shared_load, B_shared_eval);

    __syncthreads();

    for (int kk = 1; kk < k_iters; ++kk) {
        // load
        const uint32_t K_index_tile = kk * BLOCK_K * BLOCKS_Z;
        load_fp8_gds_to_lds_packed4u32<BLOCK_M, BLOCK_N, BLOCK_K, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, TF8, MatrixType::A>(
            A_shared_load, xq, M, K, M_index_tile, K_index_tile
        );
        load_fp8_gds_to_lds_packed4u32<BLOCK_M, BLOCK_N, BLOCK_K, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, TF8, MatrixType::B>(
            B_shared_load, wq, N, K, N_index_tile, K_index_tile
        );

        // compute mm
        mfma_f32_64x64x32_fp8_fp8<BLOCKS_X, BLOCKS_Y, BLOCKS_Z>(A_shared_eval, B_shared_eval, acc, A_row_stride, B_row_stride);

        // swap
        swap_ptr(A_shared_load, A_shared_eval);
        swap_ptr(B_shared_load, B_shared_eval);

        __syncthreads();
    }
    float* x_scale_shared = reinterpret_cast<float*>(A_shared_load);
    load_scale_gds_to_lds_vanilla<BLOCK_M, BLOCK_N, BLOCKS_X, BLOCKS_Y, MBLOCKS_X, MBLOCKS_Y, TF32, MatrixType::A>(x_scale, x_scale_shared, M_index_tile, M);

    // Iteration #-1 computing
    mfma_f32_64x64x32_fp8_fp8<BLOCKS_X, BLOCKS_Y, BLOCKS_Z>(A_shared_eval, B_shared_eval, acc, A_row_stride, B_row_stride);

    __syncthreads();
    // Apply scales
    apply_scale<BLOCKS_X, BLOCKS_Y, BLOCKS_Z>(x_scale_shared, w_scale, acc, N_index_tile, N);
    __syncthreads();

    // Save the result to gds in transpose to facilitate coalescence
    store_acc_to_gds_transposed<BLOCKS_X, BLOCKS_Y, TY>(y, acc, M_index_tile, N_index_tile, M, N, M);
}

template <typename FuncT>
at::Tensor f8f8bf16_rowwise_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y,
    FuncT launch_func
) {
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);

    launch_func(XQ, WQ, x_scale, w_scale, Y, M, N, K);
    return Y;
}

template <typename FuncT>
at::Tensor f8f8bf16_rowwise_wrapper(
    FuncT launch_func,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum,
    at::ScalarType out_dtype) {
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

    // Prepare output tensor if needed.
    at::Tensor Y;
    // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
    // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
    if (XQ.dim() >= 3) {
        int B = size_to_dim_(XQ.dim() - 2, XQ.sizes());
        int X_M = XQ.size(XQ.dim() - 2);
        int W_N = WQ.size(WQ.dim() - 1);
        Y = at::empty({B, W_N, X_M}, XQ.options().dtype(out_dtype));
    } else if (XQ.dim() == 2) {
        int X_M = XQ.size(XQ.dim() - 2);
        int W_N = WQ.size(WQ.dim() - 2);
        Y = at::empty({W_N, X_M}, XQ.options().dtype(out_dtype));
    } else {
        AT_ERROR("Output should at least have two dimensions");
    }

    return f8f8bf16_rowwise_impl<FuncT>(XQ, WQ, x_scale, w_scale, Y, launch_func);
}

} // namespace custom_fp8_32x32x16

#define LAUNCH_KERNEL_32x32x16(TFY, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
        { \
        dim3 grid(ceildiv(M, custom_fp8_32x32x16::BLOCK_M * BLOCKS_X * MBLOCKS_X), ceildiv(N, custom_fp8_32x32x16::BLOCK_N * BLOCKS_Y * MBLOCKS_Y), 1); \
        dim3 block(LAUNCH_WARP_SIZE, MBLOCKS_X, MBLOCKS_Y); \
        auto stream{torch::hip::getCurrentHIPStream().stream()}; \
        auto kernel = custom_fp8_32x32x16::f8f8f16_rowwise_kernel<BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, uint8_t, float, TFY>; \
        kernel<<<grid, block, 0, stream>>>( \
            reinterpret_cast<uint8_t*>(XQ.data_ptr()), \
            reinterpret_cast<uint8_t*>(WQ.data_ptr()), \
            reinterpret_cast<float*>(x_scale.data_ptr()), \
            reinterpret_cast<float*>(w_scale.data_ptr()), \
            reinterpret_cast<TFY*>(Y.data_ptr()), \
            M, N, K \
        ); \
        }

#define LAUNCH_KERNEL_OUTTYPE_32x32x16(OUT_TYPE, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
    if (OUT_TYPE == at::kFloat) { \
        LAUNCH_KERNEL_32x32x16(float, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
    } else if (OUT_TYPE == at::kHalf) { \
        LAUNCH_KERNEL_32x32x16(__half, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
    } else if (OUT_TYPE == at::kBFloat16) { \
        LAUNCH_KERNEL_32x32x16(__hip_bfloat16, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
    } else { \
        AT_ERROR("Not implemented output datatype. Must be one of {float, half, bfloat16}."); \
    }


namespace custom_fp8_16x16x32 {

constexpr uint32_t BLOCK_M = 16;
constexpr uint32_t BLOCK_N = 16;
constexpr uint32_t BLOCK_K = 32;

template <int BLOCKS_X, int BLOCKS_Y, int Z = 4>
__device__ inline uint32_t get_acc_index(uint32_t i, uint32_t j, uint32_t k = 0) {
    return (i * BLOCKS_Y + j) * Z + k;
}

// Given the lane and gpr ID, return the offset from the warp-level smem head of the 32-bit content
//   to be loaded onto the gpr. smem addresses are indexed in strides of u32
template <MatrixType MatT>
__device__ inline uint32_t get_smem_element_offset_warp_16x32_u32(
    uint32_t lane, uint32_t gpr_num, uint32_t smem_row_stride
) {
    if constexpr (MatT == MatrixType::A) {
        // A matrix
        uint32_t i = lane % 16;
        uint32_t k = 2 * (lane / 16) + gpr_num;
        return i * smem_row_stride + k;
    } else {
        // B matrix
        uint32_t i = lane % 16;
        uint32_t k = 2 * (lane / 16) + gpr_num;
        return i * smem_row_stride + k;
    }
}

template <int BLOCKS_Z>
__device__ inline void mfma_f32_16x16x32_fp8_fp8(
    const uint32_t * const A_warp_head, // indexed at warp level
    const uint32_t * const B_warp_head, // indexed at warp level
    float* acc_block,
    uint32_t A_row_stride,
    uint32_t B_row_stride
) {
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    using uint32x2 = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;
    union _reg_load {
        uint32x2 regs_;
        long long_;
    };
    // warp-level workers
#pragma unroll
    for (uint32_t k_inner_iter = 0; k_inner_iter < BLOCKS_Z; ++k_inner_iter) {
        const uint32_t* A_warp_head_inner = A_warp_head + k_inner_iter * (BLOCK_K / 4);
        const uint32_t* B_warp_head_inner = B_warp_head + k_inner_iter * (BLOCK_K / 4);
        _reg_load a_regs;
        _reg_load b_regs;
#pragma unroll
        for (uint32_t reg = 0; reg < 2; ++reg) {
            a_regs.regs_[reg] = A_warp_head_inner[get_smem_element_offset_warp_16x32_u32<MatrixType::A>(threadIdx.x, reg, A_row_stride)];
            b_regs.regs_[reg] = B_warp_head_inner[get_smem_element_offset_warp_16x32_u32<MatrixType::B>(threadIdx.x, reg, B_row_stride)];
        }
        floatx4* acc_block_f4 = reinterpret_cast<floatx4*>(acc_block);
        *acc_block_f4 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_regs.long_, b_regs.long_, *acc_block_f4, 0, 0, 0);
    }
}

template <
    uint32_t BLOCKS_X, uint32_t BLOCKS_Y, uint32_t BLOCKS_Z
>
__device__ inline void mfma_tb_f32_16x16x32_fp8_fp8(
    const uint32_t * const A_head_tb, // indexed at threadblock level
    const uint32_t * const B_head_tb, // indexed at threadblock level
    float* acc,
    uint32_t A_row_stride, // in u32
    uint32_t B_row_stride  // in u32
) {
    uint32_t warp_x = threadIdx.y;
    uint32_t warp_y = threadIdx.z;
#pragma unroll
    for (uint32_t x_iter = 0; x_iter < BLOCKS_X; ++x_iter) {
        uint32_t x_load_block = warp_x * BLOCKS_X + x_iter;
        const uint32_t* A_warp_head = A_head_tb + x_load_block * BLOCK_M * A_row_stride;
#pragma unroll
        for (uint32_t y_iter = 0; y_iter < BLOCKS_Y; ++y_iter) {
            uint32_t y_load_block = warp_y * BLOCKS_Y + y_iter;
            const uint32_t* B_warp_head = B_head_tb + y_load_block * BLOCK_N * B_row_stride;
            float* acc_block = acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 4>(x_iter, y_iter);
            mfma_f32_16x16x32_fp8_fp8<BLOCKS_Z>(A_warp_head, B_warp_head, acc_block, A_row_stride, B_row_stride);
        }
    }
}

template <int BLOCKS_X, int BLOCKS_Y, int MBLOCKS_X, int MBLOCKS_Y>
__device__ inline void apply_scale_16x16x32(
    const float * const lds_xscale,
    const float * const lds_wscale,
    float* acc,
    uint32_t M_index_tile,
    uint32_t N_index_tile,
    uint32_t M,
    uint32_t N
) {
    const uint32_t lane_id = threadIdx.x;
    const uint32_t xscale_warp_offset = threadIdx.y * BLOCKS_X * BLOCK_M;
    const uint32_t wscale_warp_offset = threadIdx.z * BLOCKS_Y * BLOCK_N;
#pragma unroll
    for (uint32_t iter_y = 0; iter_y < BLOCKS_Y; ++iter_y) {
        const uint32_t wscale_elem_offset = wscale_warp_offset + iter_y * BLOCK_N + (lane_id % 16);
        float wscale = (N_index_tile + wscale_elem_offset < N) ? lds_wscale[wscale_elem_offset] : 0.0f;
#pragma unroll
        for (uint32_t iter_x = 0; iter_x < BLOCKS_X; ++iter_x) {
            float* acc_warp = acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 4>(iter_x, iter_y);
            const uint32_t xscale_iter_offset = xscale_warp_offset + iter_x * BLOCK_M;
            for (uint32_t reg = 0; reg < 4; ++reg) {
                const uint32_t xscale_elem_offset = xscale_iter_offset + 4 * (lane_id / 16) + (reg % 4);
                acc_warp[reg] *= wscale;
                acc_warp[reg] *= (M_index_tile + xscale_elem_offset < M) ? lds_xscale[xscale_elem_offset] : 0.0f;
            }
        }
    }
}

template <int BLOCKS_X, int BLOCKS_Y, typename TY>
__device__ inline void store_acc_to_gds_transposed_16x16(
    TY* y_gds,
    float* acc,
    uint32_t M_head_index, // col head id
    uint32_t N_head_index, // row head id
    uint32_t M,
    uint32_t N,
    uint32_t y_row_stride // should be M
) {
    const uint32_t lane_id = threadIdx.x;
    const uint32_t M_warp_head_offset = M_head_index + threadIdx.y * BLOCK_M * BLOCKS_X;
    const uint32_t N_warp_head_offset = N_head_index + threadIdx.z * BLOCK_N * BLOCKS_Y;
#pragma unroll
    for (uint32_t warp_m = 0; warp_m < BLOCKS_X; ++warp_m) {
        const uint32_t M_warp_iter_head_offset = M_warp_head_offset + warp_m * BLOCK_M;
#pragma unroll
        for (uint32_t warp_n = 0; warp_n < BLOCKS_Y; ++warp_n) {
            const uint32_t N_warp_iter_head_offset = N_warp_head_offset + warp_n * BLOCK_N;
            const uint32_t N_lane_reg_offset = N_warp_iter_head_offset + (lane_id % 16);
            if (N_lane_reg_offset >= N) { continue; }
            const uint32_t N_offset_strided = N_lane_reg_offset * y_row_stride;

            const uint32_t M_lane_reg_offset = M_warp_iter_head_offset + 4 * (lane_id / 16); // + reg
            if (M_lane_reg_offset >= M) { continue; }
            const uint32_t num_items = M - M_lane_reg_offset;
            if (num_items > 3) {
                store_regs4_to_gds<TY, 4>(y_gds + N_offset_strided + M_lane_reg_offset, acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 4>(warp_m, warp_n));
            } else if (num_items == 3) {
                store_regs4_to_gds<TY, 3>(y_gds + N_offset_strided + M_lane_reg_offset, acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 4>(warp_m, warp_n));
            } else if (num_items == 2) {
                store_regs4_to_gds<TY, 2>(y_gds + N_offset_strided + M_lane_reg_offset, acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 4>(warp_m, warp_n));
            } else {
                store_regs4_to_gds<TY, 1>(y_gds + N_offset_strided + M_lane_reg_offset, acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 4>(warp_m, warp_n));
            }
        }
    }
}

template <int BLOCKS_X, int BLOCKS_Y, typename TY>
__device__ inline void store_acc_to_gds_transposed_16x16_packed4u32(
    TY* y_gds,
    float* acc,
    uint32_t M_head_index, // col head id
    uint32_t N_head_index, // row head id
    uint32_t M,
    uint32_t N,
    uint32_t y_row_stride // should be M
) {
    if constexpr (sizeof(TY) == 4 || (((BLOCKS_X * BLOCKS_Y) % 2) == 1)) {
        store_acc_to_gds_transposed_16x16<BLOCKS_X, BLOCKS_Y, TY>(y_gds, acc, M_head_index, N_head_index, M, N, y_row_stride);
        return; 
    } else {
        // For sizeof(TY) == 2 and BLOCKS_X * BLOCKS_Y = 2x
        static_assert(sizeof(TY) == 2);
        static_assert((BLOCKS_X * BLOCKS_Y) % 2 == 0);
        const uint32_t lane_id = threadIdx.x;
        const uint32_t M_warp_head_offset = M_head_index + threadIdx.y * BLOCK_M * BLOCKS_X;
        const uint32_t N_warp_head_offset = N_head_index + threadIdx.z * BLOCK_N * BLOCKS_Y;

        const bool second_block = ((threadIdx.x / 16) % 2 == 1);

        using uint32x2 = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;
        using uint32x4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
        union tx_buffer_t {
            TY f16[8];
            uint32x2 u64[2];
        };
        union ex_buffer_t {
            TY f16[4];
            unsigned long long u64;
        };

        constexpr uint32_t num_blocks = BLOCKS_X * BLOCKS_Y;
        constexpr uint32_t num_iters = ceildiv(num_blocks, 2);

#pragma unroll
        for (uint32_t block_iter = 0; block_iter < num_iters; ++block_iter) {
            const uint32_t block_id_flattened_tx = block_iter * 2 + second_block;
            const uint32_t block_id_flattened_ex = block_iter * 2 + (!second_block);
            const uint32_t warp_n = block_id_flattened_tx / BLOCKS_X;
            const uint32_t warp_m = block_id_flattened_tx % BLOCKS_X;
            const uint32_t warp_n_ex = block_id_flattened_ex / BLOCKS_X;
            const uint32_t warp_m_ex = block_id_flattened_ex % BLOCKS_X;

            const uint32_t N_warp_iter_head_offset = N_warp_head_offset + warp_n * BLOCK_N;
            const uint32_t M_warp_iter_head_offset = M_warp_head_offset + warp_m * BLOCK_M;

            ex_buffer_t ex_buffer;
#pragma unroll
            for (size_t i = 0; i < 4; ++i) {
                (ex_buffer.f16)[i] = static_cast<TY>(*(acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 4>(warp_m_ex, warp_n_ex, i)));
            } 
            ex_buffer.u64 = __shfl_xor(ex_buffer.u64, 0x00000010);

            tx_buffer_t tx_buffer;
            TY* tx_buffer_ty = reinterpret_cast<TY*>(&tx_buffer);
            const uint32_t tx_buffer_offset_native = 4 * second_block;
            const uint32_t tx_buffer_offset_transf = 4 * (!second_block);
#pragma unroll
            for (size_t i = 0; i < 4; ++i) {
                tx_buffer.f16[tx_buffer_offset_native + i] = static_cast<TY>(*(acc + get_acc_index<BLOCKS_X, BLOCKS_Y, 4>(warp_m, warp_n, i)));
                tx_buffer.f16[tx_buffer_offset_transf + i] = ex_buffer.f16[i];
            }

            const uint32_t N_lane_reg_offset = N_warp_iter_head_offset + (lane_id % 16);
            if (N_lane_reg_offset >= N) { continue; }

            const uint32_t M_lane_reg_offset = M_warp_iter_head_offset + 8 * (lane_id / 32);
            if (M_lane_reg_offset >= M) { continue; }

            const int num_items = (M - M_lane_reg_offset) > 8 ? 8 : (M - M_lane_reg_offset);
            int tx_offset = 0; // in steps of type TY
            while (tx_offset < num_items) {
                const int remaining_items = num_items - tx_offset;
                if (remaining_items >= 8) {
                    store_regs4_to_gds<float, 4>(reinterpret_cast<float*>(y_gds + N_lane_reg_offset * y_row_stride + M_lane_reg_offset + tx_offset), reinterpret_cast<const float*>(tx_buffer_ty + tx_offset));
                    tx_offset += 8;
                } else if (remaining_items >= 4) {
                    store_regs4_to_gds<float, 2>(reinterpret_cast<float*>(y_gds + N_lane_reg_offset * y_row_stride + M_lane_reg_offset + tx_offset), reinterpret_cast<const float*>(tx_buffer_ty + tx_offset));
                    tx_offset += 4;
                } else if (remaining_items >= 2) {
                    store_regs4_to_gds<float, 1>(reinterpret_cast<float*>(y_gds + N_lane_reg_offset * y_row_stride + M_lane_reg_offset + tx_offset), reinterpret_cast<const float*>(tx_buffer_ty + tx_offset));
                    tx_offset += 2;
                } else {
                    float item_f32 = static_cast<float>(*(tx_buffer_ty + tx_offset));
                    store_regs4_to_gds<TY, 1>(y_gds + N_lane_reg_offset * y_row_stride + M_lane_reg_offset + tx_offset, &item_f32);
                    tx_offset += 1;
                }
            }
        }
    }
}

template <
    uint32_t BLOCKS_X, uint32_t BLOCKS_Y, uint32_t BLOCKS_Z,
    uint32_t MBLOCKS_X, uint32_t MBLOCKS_Y,
    typename TF8, typename TF32, typename TY
>
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
    constexpr uint32_t TBLOCKS_M = BLOCK_M * BLOCKS_X;
    constexpr uint32_t TBLOCKS_N = BLOCK_N * BLOCKS_Y;
    constexpr uint32_t TBLOCKS_K = BLOCK_K * BLOCKS_Z;
    constexpr uint32_t MBLOCKS_M = TBLOCKS_M * MBLOCKS_X;
    constexpr uint32_t MBLOCKS_N = TBLOCKS_N * MBLOCKS_Y;

    constexpr uint32_t A_block_size_u32 = BLOCK_M * BLOCK_K / (4 / sizeof(TF8));
    constexpr uint32_t A_warp_block_size_u32 = A_block_size_u32 * BLOCKS_X * BLOCKS_Z;
    constexpr uint32_t A_tile_block_size_u32 = A_warp_block_size_u32 * MBLOCKS_X;
    constexpr uint32_t B_block_size_u32 = BLOCK_N * BLOCK_K / (4 / sizeof(TF8));
    constexpr uint32_t B_warp_block_size_u32 = B_block_size_u32 * BLOCKS_Y * BLOCKS_Z;
    constexpr uint32_t B_tile_block_size_u32 = B_warp_block_size_u32 * MBLOCKS_Y;

    __shared__ __attribute__((aligned(512))) uint32_t A_shared[A_tile_block_size_u32 * 2];
    __shared__ __attribute__((aligned(512))) uint32_t B_shared[B_tile_block_size_u32 * 2]; // transpoesd

    constexpr uint32_t A_row_stride = BLOCK_K / (4 / sizeof(TF8)) * BLOCKS_Z;
    constexpr uint32_t B_row_stride = BLOCK_K / (4 / sizeof(TF8)) * BLOCKS_Z;

    uint32_t* A_shared_load = A_shared;
    uint32_t* A_shared_eval = A_shared + A_tile_block_size_u32;
    uint32_t* B_shared_load = B_shared;
    uint32_t* B_shared_eval = B_shared + B_tile_block_size_u32;

    initialize_smem(A_shared_load, A_tile_block_size_u32);
    initialize_smem(B_shared_load, B_tile_block_size_u32);

    float acc[BLOCKS_X * BLOCKS_Y * 4];
#pragma unroll
    for (auto i = 0; i < BLOCKS_X * BLOCKS_Y * 4; ++i) {
        acc[i] = 0.0f;
    }

    const uint32_t M_index_tile = blockIdx.x * MBLOCKS_M; // head of threadblock
    const uint32_t N_index_tile = blockIdx.y * MBLOCKS_N; // head of threadblock

    const uint32_t k_iters = ceildiv(K, BLOCK_K * BLOCKS_Z);

    __syncthreads();

    load_fp8_gds_to_lds_packed4u32<BLOCK_M, BLOCK_N, BLOCK_K, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, TF8, MatrixType::A>(
        A_shared_load, xq, M, K, M_index_tile, 0);
    load_fp8_gds_to_lds_packed4u32<BLOCK_M, BLOCK_N, BLOCK_K, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, TF8, MatrixType::B>(
        B_shared_load, wq, N, K, N_index_tile, 0);

    swap_ptr(A_shared_load, A_shared_eval);
    swap_ptr(B_shared_load, B_shared_eval);

    __syncthreads();

    for (int kk = 1; kk < k_iters; ++kk) {
        const uint32_t K_index_tile = kk * BLOCK_K * BLOCKS_Z;

        load_fp8_gds_to_lds_packed4u32<BLOCK_M, BLOCK_N, BLOCK_K, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, TF8, MatrixType::A>(
            A_shared_load, xq, M, K, M_index_tile, K_index_tile);
        load_fp8_gds_to_lds_packed4u32<BLOCK_M, BLOCK_N, BLOCK_K, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, TF8, MatrixType::B>(
            B_shared_load, wq, N, K, N_index_tile, K_index_tile);

        mfma_tb_f32_16x16x32_fp8_fp8<BLOCKS_X, BLOCKS_Y, BLOCKS_Z>(
            A_shared_eval, B_shared_eval, acc, A_row_stride, B_row_stride
        );

        swap_ptr(A_shared_load, A_shared_eval);
        swap_ptr(B_shared_load, B_shared_eval);

        __syncthreads();
    }

    float* x_scale_shared = reinterpret_cast<float*>(A_shared_load);
    float* w_scale_shared = reinterpret_cast<float*>(B_shared_load);
    load_scale_gds_to_lds_vanilla<BLOCK_M, BLOCK_N, BLOCKS_X, BLOCKS_Y, MBLOCKS_X, MBLOCKS_Y, TF32, MatrixType::A>(
        x_scale, x_scale_shared, M_index_tile, M
    );
    load_scale_gds_to_lds_vanilla<BLOCK_M, BLOCK_N, BLOCKS_X, BLOCKS_Y, MBLOCKS_X, MBLOCKS_Y, TF32, MatrixType::B>(
        w_scale, w_scale_shared, N_index_tile, N
    );

    mfma_tb_f32_16x16x32_fp8_fp8<BLOCKS_X, BLOCKS_Y, BLOCKS_Z>(
        A_shared_eval, B_shared_eval, acc, A_row_stride, B_row_stride
    );
    
    __syncthreads();
    apply_scale_16x16x32<BLOCKS_X, BLOCKS_Y, MBLOCKS_X, MBLOCKS_Y>(
        x_scale_shared, w_scale_shared, acc, M_index_tile, N_index_tile, M, N
    );

    __syncthreads();

    store_acc_to_gds_transposed_16x16<BLOCKS_X, BLOCKS_Y>(y, acc, M_index_tile, N_index_tile, M, N, M);
    // store_acc_to_gds_transposed_16x16_packed4u32<BLOCKS_X, BLOCKS_Y>(y, acc, M_index_tile, N_index_tile, M, N, M);
}

template <typename FuncT>
at::Tensor f8f8bf16_rowwise_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y,
    FuncT launch_func
) {
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);

    launch_func(XQ, WQ, x_scale, w_scale, Y, M, N, K);
    return Y;
}

template <typename FuncT>
at::Tensor f8f8bf16_rowwise_wrapper(
    FuncT launch_func,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum,
    at::ScalarType out_dtype) {
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

    // Prepare output tensor if needed.
    at::Tensor Y;
    // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
    // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
    if (XQ.dim() >= 3) {
        int B = size_to_dim_(XQ.dim() - 2, XQ.sizes());
        int X_M = XQ.size(XQ.dim() - 2);
        int W_N = WQ.size(WQ.dim() - 1);
        Y = at::empty({B, W_N, X_M}, XQ.options().dtype(out_dtype));
    } else if (XQ.dim() == 2) {
        int X_M = XQ.size(XQ.dim() - 2);
        int W_N = WQ.size(WQ.dim() - 2);
        Y = at::empty({W_N, X_M}, XQ.options().dtype(out_dtype));
    } else {
        AT_ERROR("Output should at least have two dimensions");
    }

    return f8f8bf16_rowwise_impl<FuncT>(XQ, WQ, x_scale, w_scale, Y, launch_func);
}

} // namespace custom_fp8_16x16x32

#define LAUNCH_KERNEL_16x16x32(TFY, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
        { \
        dim3 grid(ceildiv(M, custom_fp8_16x16x32::BLOCK_M * BLOCKS_X * MBLOCKS_X), ceildiv(N, custom_fp8_16x16x32::BLOCK_N * BLOCKS_Y * MBLOCKS_Y), 1); \
        dim3 block(LAUNCH_WARP_SIZE, MBLOCKS_X, MBLOCKS_Y); \
        auto stream{torch::hip::getCurrentHIPStream().stream()}; \
        auto kernel = custom_fp8_16x16x32::f8f8f16_rowwise_kernel<BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, uint8_t, float, TFY>; \
        kernel<<<grid, block, 0, stream>>>( \
            reinterpret_cast<uint8_t*>(XQ.data_ptr()), \
            reinterpret_cast<uint8_t*>(WQ.data_ptr()), \
            reinterpret_cast<float*>(x_scale.data_ptr()), \
            reinterpret_cast<float*>(w_scale.data_ptr()), \
            reinterpret_cast<TFY*>(Y.data_ptr()), \
            M, N, K \
        ); \
        }

#define LAUNCH_KERNEL_OUTTYPE_16x16x32(OUT_TYPE, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
    if (OUT_TYPE == at::kFloat) { \
        LAUNCH_KERNEL_16x16x32(float, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
    } else if (OUT_TYPE == at::kHalf) { \
        LAUNCH_KERNEL_16x16x32(__half, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
    } else if (OUT_TYPE == at::kBFloat16) { \
        LAUNCH_KERNEL_16x16x32(__hip_bfloat16, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K) \
    } else { \
        AT_ERROR("Not implemented output datatype. Must be one of {float, half, bfloat16}."); \
    }
