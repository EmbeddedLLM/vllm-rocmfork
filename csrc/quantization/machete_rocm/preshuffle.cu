#include "ops.cuh"
#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <functional>
#include <ATen/ATen.h>
#ifdef USE_ROCM
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAStream.h>
#endif
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/literals.hpp"
#include "utils.hpp"

#define KPack 16
#define KLane 64 / NLane
#define ThreadsPerBlock 256

using FP8  = ck::f8_t;

namespace {

  int __host__ __device__ compute_lane_index(
    const int n,
    const int k,
    const int K,
    const int NXdl
  ){
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    const int NLane = NXdl;
    const int K0 = K / (KLane * KPack);
    const int n0 = n / NLane;
    const int n1 = n % NLane;
    const int k0 = k / (KLane * KPack);
    const int tempk = k % (KLane * KPack);
    const int k1 = tempk / KPack;
    const int k2 = tempk % KPack;

    return n0 * KPack * NLane * KLane * K0 + 
            k0 * KPack * NLane * KLane      +
            k1 * KPack * NLane              +
            n1 * KPack                      + 
            k2;
  }

  template <typename scalar_t>
  static __global__ void preshuffle_kernel(
    const scalar_t* source,
    scalar_t* destination,
    const int N,
    const int K,
    const int NXdl
  ){
      const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total_threads = gridDim.x * blockDim.x;
      for (int i = thread_idx; i < N * K; i += total_threads) {
        int n = i / K;
        int k = i % K;
        if(n < N && k<K)
          destination[compute_lane_index(n, k, K, NXdl)] = source[n * K + k];
      }
  }

  void preshuffle_cpu_launcher(
    const FP8* source_ptr,
    FP8* destination_ptr,
    const int N, 
    const int K,
    const int NXdl
  ){
      for(int n = 0; n < N; ++n)
        for(int k = 0; k < K; ++k)
          destination_ptr[compute_lane_index(n, k, K, NXdl)] = source_ptr[n * K + k];
  }

  void preshuffle_launcher(
    const FP8* source_ptr, 
    FP8* destination_ptr, 
    const int N, 
    const int K, 
    const int NXdl
  ){
    
    const int total_elements = N * K;
    const int blocks = (total_elements + ThreadsPerBlock - 1) / ThreadsPerBlock;
    auto stream = at::cuda::getCurrentHIPStream().stream();

    preshuffle_kernel<FP8><<<blocks, ThreadsPerBlock, 0, stream>>>(
      source_ptr,
      destination_ptr,
      N,
      K,
      NXdl
    );

    const auto cuda_last_error = cudaGetLastError();
    if (cudaSuccess != cuda_last_error)
      throw std::runtime_error("CUDA kernel failed : " + std::to_string(cuda_last_error));
              
  }

  at::Tensor preshuffle_impl(
    const at::Tensor& source,
    const int NXdl,
    const std::function<void(const FP8*, FP8*, const int, const int, const int)>& launcher
  ){
    const auto source_aligned = machete_rocm::utils::align_to_wavefront(source);
    const FP8* source_ptr = reinterpret_cast<FP8*>(source_aligned.data_ptr());

    const auto N = source_aligned.size(0);
    const auto K = source_aligned.size(1);
    
    auto destination = torch::zeros(source_aligned.sizes(), source_aligned.options());
    FP8* destination_ptr = reinterpret_cast<FP8*>(destination.data_ptr());

    launcher(source_ptr, destination_ptr, N, K, NXdl);
    return destination;
  }

}

at::Tensor machete_rocm::preshuffle(const at::Tensor tensor, const int64_t NXdl) {
  return preshuffle_impl(tensor, NXdl, preshuffle_launcher);
}

at::Tensor machete_rocm::preshuffle_cpu(const at::Tensor tensor, const int64_t NXdl) {
  return preshuffle_impl(tensor, NXdl, preshuffle_cpu_launcher);
}

