#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <ATen/ATen.h>
#ifdef USE_ROCM
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp"
// #include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

// #include "ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle.hpp"

// #include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
// #include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
// #include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"


template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using F32  = float;

namespace machete {

template <typename InOutDataType>
void preShuffleBufferCPU(const InOutDataType* src, InOutDataType* dst, int N, int K, int NXdl)
{
    int KPack = 16;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    // printf("CPU, outputIndex, sourceIndex\n");
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K + k];

        }
    }
}



template <typename Element>
static __global__ void preShuffleBufferKernel(const Element* src, Element* dst, int N, int K, int NXdl) {
    const int KPack = 16;
    const int NLane = NXdl;
    const int KLane = 64 / NLane;
    const int K0 = K / (KLane * KPack);

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    // printf("GPU, outputIndex, sourceIndex\n");

    for (int idx = global_idx; idx < N * K; idx += total_threads) {
        int n = idx / K;
        int k = idx % K;

        if (n < N && k < K)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            int tempk = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                            k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[idx];  // This is equivalent to src[n * K + k]
            // dst[outputIndex] = src[n * K + k];  // This is equivalent to src[n * K + k]
        }
    }
}



template <typename Element>
static void prepackB_launcher(cudaStream_t stream, const Element* src, Element* dst, int N, int K, int NXdl) {
    // Define the number of threads per block
    const int threads_per_block = 256;

    // Calculate the number of blocks needed
    int total_elements = N * K;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    machete::preShuffleBufferKernel<Element><<<blocks, threads_per_block, 0, stream>>>(src, dst, N, K, NXdl);
}

} // namespace machete


at::Tensor prepackB(at::Tensor B, const int NXdl=32) {

    TORCH_CHECK(
        (B.dtype() == at::kFloat8_e4m3fnuz)),
        "Inputs must be type float8_e4m3fnuz.");

    const int N = B.size(0);
    const int K = B.size(1);

    // Allocate output
    torch::Tensor Bprepacked = torch::empty_like(B, {}, at::MemoryFormat::Contiguous);
    auto stream = at::cuda::getCurrentHIPStream().stream();

    machete::prepackB_launcher(stream, 
        reinterpret_cast<FP8*>(B.data_ptr()), 
        reinterpret_cast<FP8*>(Bprepacked.mutable_data_ptr()), 
        N, K, NXdl
    );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

at::Tensor prepackB_cpu(at::Tensor B, const int NXdl=32) {

    TORCH_CHECK(
        (B.dtype() == at::kFloat8_e4m3fnuz),
        "Inputs must be type float8_e4m3fnuz.");
        
    const int N = B.size(0);
    const int K = B.size(1);

    // Allocate output
    torch::Tensor Bprepacked = torch::empty_like(B, {}, at::MemoryFormat::Contiguous);
    auto B_ptr = reinterpret_cast<FP8*>(B.data_ptr());
    auto Bprepacked_ptr = reinterpret_cast<FP8*>(Bprepacked.mutable_data_ptr());
    machete::preShuffleBufferCPU(B_ptr, Bprepacked_ptr, N, K, NXdl);

}