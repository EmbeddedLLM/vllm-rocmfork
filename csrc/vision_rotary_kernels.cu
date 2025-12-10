#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <type_traits>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename T, int Width>
struct alignas(16) AlignedVec {
  T val[Width];
};

// Kernel for applying rotary embeddings to vision transformers
// This kernel processes both q and k tensors simultaneously using precomputed cos/sin
// Input tensors: q, k of shape [batch, seqlen, nheads, headdim]
// cos, sin of shape [seqlen, rotary_dim/2] (will be broadcast across batch and heads)
template <typename scalar_t>
__global__ void apply_vision_rotary_2c_kernel(
    scalar_t* __restrict__ out_q,          // [batch, seqlen, nheads, headdim]
    scalar_t* __restrict__ out_k,          // [batch, seqlen, nheads, headdim]
    const scalar_t* __restrict__ in_q,     // [batch, seqlen, nheads, headdim]
    const scalar_t* __restrict__ in_k,     // [batch, seqlen, nheads, headdim]
    const scalar_t* __restrict__ cos,      // [seqlen, rotary_dim / 2]
    const scalar_t* __restrict__ sin,      // [seqlen, rotary_dim / 2]
    const int64_t batch_size,
    const int64_t seqlen,
    const int64_t nheads,
    const int64_t headdim,
    const int64_t rotary_dim,
    const int64_t stride_q_batch,
    const int64_t stride_q_seqlen,
    const int64_t stride_q_nheads,
    const int64_t stride_q_headdim,
    const int64_t stride_k_batch,
    const int64_t stride_k_seqlen,
    const int64_t stride_k_nheads,
    const int64_t stride_k_headdim,
    const int64_t stride_cos_seqlen,
    const int64_t stride_cos_dim) {
  
  const int64_t rotary_dim_half = rotary_dim / 2;
  
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = batch_size * seqlen * nheads * rotary_dim_half;
  
  if (idx >= total_elements) return;
  
  const int64_t dim_idx = idx % rotary_dim_half;
  const int64_t tmp1 = idx / rotary_dim_half;
  const int64_t head_idx = tmp1 % nheads;
  const int64_t tmp2 = tmp1 / nheads;
  const int64_t seq_idx = tmp2 % seqlen;
  const int64_t batch_idx = tmp2 / seqlen;
  
  // Compute offsets for q tensor
  const int64_t in_q_offset = batch_idx * stride_q_batch + 
                               seq_idx * stride_q_seqlen + 
                               head_idx * stride_q_nheads + 
                               dim_idx * stride_q_headdim;
  
  // Compute offsets for k tensor
  const int64_t in_k_offset = batch_idx * stride_k_batch + 
                               seq_idx * stride_k_seqlen + 
                               head_idx * stride_k_nheads + 
                               dim_idx * stride_k_headdim;
  
  // Load q values (first half and second half)
  const float q0 = static_cast<float>(VLLM_LDG(&in_q[in_q_offset]));
  const float k0 = static_cast<float>(VLLM_LDG(&in_k[in_k_offset]));
  
  const int64_t in_q_offset_half = in_q_offset + rotary_dim_half * stride_q_headdim;
  const int64_t in_k_offset_half = in_k_offset + rotary_dim_half * stride_k_headdim;
  
  const float q1 = static_cast<float>(VLLM_LDG(&in_q[in_q_offset_half]));
  const float k1 = static_cast<float>(VLLM_LDG(&in_k[in_k_offset_half]));
  
  // Load precomputed cos and sin values
  const int64_t cos_sin_offset = seq_idx * stride_cos_seqlen + dim_idx * stride_cos_dim;
  const float cos_val = static_cast<float>(VLLM_LDG(&cos[cos_sin_offset]));
  const float sin_val = static_cast<float>(VLLM_LDG(&sin[cos_sin_offset]));
  
  // Apply rotary embedding to q
  const scalar_t out_q0 = static_cast<scalar_t>(q0 * cos_val - q1 * sin_val);
  const scalar_t out_q1 = static_cast<scalar_t>(q0 * sin_val + q1 * cos_val);
  
  // Apply rotary embedding to k
  const scalar_t out_k0 = static_cast<scalar_t>(k0 * cos_val - k1 * sin_val);
  const scalar_t out_k1 = static_cast<scalar_t>(k0 * sin_val + k1 * cos_val);
  
  // Store results
  out_q[in_q_offset] = out_q0;
  out_q[in_q_offset_half] = out_q1;
  out_k[in_k_offset] = out_k0;
  out_k[in_k_offset_half] = out_k1;
}

// Vectorized variant that processes a full 16B chunk per transaction.
// Vector width adapts to scalar type (float/bfloat16/half).
template <typename scalar_t>
__global__ void apply_vision_rotary_2c_vec_kernel(
    scalar_t* __restrict__ out_q,          // [batch, seqlen, nheads, headdim]
    scalar_t* __restrict__ out_k,          // [batch, seqlen, nheads, headdim]
    const scalar_t* __restrict__ in_q,     // [batch, seqlen, nheads, headdim]
    const scalar_t* __restrict__ in_k,     // [batch, seqlen, nheads, headdim]
    const scalar_t* __restrict__ cos,      // [seqlen, rotary_dim / 2]
    const scalar_t* __restrict__ sin,      // [seqlen, rotary_dim / 2]
    const int64_t batch_size,
    const int64_t seqlen,
    const int64_t nheads,
    const int64_t headdim,
    const int64_t rotary_dim,
    const int64_t stride_q_batch,
    const int64_t stride_q_seqlen,
    const int64_t stride_q_nheads,
    const int64_t stride_q_headdim,
    const int64_t stride_k_batch,
    const int64_t stride_k_seqlen,
    const int64_t stride_k_nheads,
    const int64_t stride_k_headdim,
    const int64_t stride_cos_seqlen,
    const int64_t stride_cos_dim) {

  static_assert((16 % sizeof(scalar_t)) == 0, "Vector width must divide 16B");
  constexpr int kVecWidth = 16 / sizeof(scalar_t);
  const int64_t rotary_dim_half = rotary_dim / 2;
  const int64_t rotary_vecs = rotary_dim_half / kVecWidth;

  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_vec = batch_size * seqlen * nheads * rotary_vecs;
  if (idx >= total_vec) return;

  using Vec = AlignedVec<scalar_t, kVecWidth>;

  const int64_t vec_offset = idx % rotary_vecs;
  const int64_t tmp1 = idx / rotary_vecs;
  const int64_t head_idx = tmp1 % nheads;
  const int64_t tmp2 = tmp1 / nheads;
  const int64_t seq_idx = tmp2 % seqlen;
  const int64_t batch_idx = tmp2 / seqlen;

  const int64_t dim_idx = vec_offset * kVecWidth;

  const int64_t in_q_base = batch_idx * stride_q_batch +
                            seq_idx * stride_q_seqlen +
                            head_idx * stride_q_nheads +
                            dim_idx * stride_q_headdim;

  const int64_t in_k_base = batch_idx * stride_k_batch +
                            seq_idx * stride_k_seqlen +
                            head_idx * stride_k_nheads +
                            dim_idx * stride_k_headdim;

  const scalar_t* q_ptr = in_q + in_q_base;
  const scalar_t* k_ptr = in_k + in_k_base;

  const Vec q0_vec = *reinterpret_cast<const Vec*>(q_ptr);
  const Vec k0_vec = *reinterpret_cast<const Vec*>(k_ptr);

  const scalar_t* q_ptr_half = q_ptr + rotary_dim_half * stride_q_headdim;
  const scalar_t* k_ptr_half = k_ptr + rotary_dim_half * stride_k_headdim;

  const Vec q1_vec = *reinterpret_cast<const Vec*>(q_ptr_half);
  const Vec k1_vec = *reinterpret_cast<const Vec*>(k_ptr_half);

  const int64_t cos_sin_offset = seq_idx * stride_cos_seqlen +
                                 dim_idx * stride_cos_dim;
  const Vec cos_vec = *reinterpret_cast<const Vec*>(&cos[cos_sin_offset]);
  const Vec sin_vec = *reinterpret_cast<const Vec*>(&sin[cos_sin_offset]);

  Vec out_q0_vec, out_q1_vec, out_k0_vec, out_k1_vec;
  #pragma unroll
  for (int i = 0; i < kVecWidth; ++i) {
    const float q0 = static_cast<float>(q0_vec.val[i]);
    const float q1 = static_cast<float>(q1_vec.val[i]);
    const float k0 = static_cast<float>(k0_vec.val[i]);
    const float k1 = static_cast<float>(k1_vec.val[i]);
    const float cos_val = static_cast<float>(cos_vec.val[i]);
    const float sin_val = static_cast<float>(sin_vec.val[i]);

    out_q0_vec.val[i] = static_cast<scalar_t>(q0 * cos_val - q1 * sin_val);
    out_q1_vec.val[i] = static_cast<scalar_t>(q0 * sin_val + q1 * cos_val);
    out_k0_vec.val[i] = static_cast<scalar_t>(k0 * cos_val - k1 * sin_val);
    out_k1_vec.val[i] = static_cast<scalar_t>(k0 * sin_val + k1 * cos_val);
  }

  scalar_t* out_q_ptr = out_q + in_q_base;
  scalar_t* out_k_ptr = out_k + in_k_base;

  *reinterpret_cast<Vec*>(out_q_ptr) = out_q0_vec;
  *reinterpret_cast<Vec*>(out_k_ptr) = out_k0_vec;

  scalar_t* out_q_ptr_half = out_q_ptr + rotary_dim_half * stride_q_headdim;
  scalar_t* out_k_ptr_half = out_k_ptr + rotary_dim_half * stride_k_headdim;

  *reinterpret_cast<Vec*>(out_q_ptr_half) = out_q1_vec;
  *reinterpret_cast<Vec*>(out_k_ptr_half) = out_k1_vec;
}

}  // namespace vllm

void apply_vision_rotary_2c(
    torch::Tensor& out_q,           // [batch, seqlen, nheads, headdim]
    torch::Tensor& out_k,           // [batch, seqlen, nheads, headdim]
    const torch::Tensor& in_q,      // [batch, seqlen, nheads, headdim]
    const torch::Tensor& in_k,      // [batch, seqlen, nheads, headdim]
    const torch::Tensor& cos,       // [seqlen, rotary_dim / 2]
    const torch::Tensor& sin,       // [seqlen, rotary_dim / 2]
    int64_t rotary_dim) {
  
  const int64_t batch_size = in_q.size(0);
  const int64_t seqlen = in_q.size(1);
  const int64_t nheads = in_q.size(2);
  const int64_t headdim = in_q.size(3);
  
  TORCH_CHECK(in_q.dim() == 4, "in_q must be 4D tensor");
  TORCH_CHECK(in_k.dim() == 4, "in_k must be 4D tensor");
  TORCH_CHECK(cos.dim() == 2, "cos must be 2D tensor");
  TORCH_CHECK(sin.dim() == 2, "sin must be 2D tensor");
  TORCH_CHECK(in_q.sizes() == in_k.sizes(), "in_q and in_k must have same shape");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have same shape");
  TORCH_CHECK(cos.size(1) * 2 == rotary_dim, "cos second dim must be rotary_dim / 2");
  TORCH_CHECK(rotary_dim <= headdim, "rotary_dim must be <= headdim");
  TORCH_CHECK(cos.size(0) >= seqlen, "cos first dim must be >= seqlen");
  
  const int64_t stride_q_batch = in_q.stride(0);
  const int64_t stride_q_seqlen = in_q.stride(1);
  const int64_t stride_q_nheads = in_q.stride(2);
  const int64_t stride_q_headdim = in_q.stride(3);
  const int64_t stride_k_batch = in_k.stride(0);
  const int64_t stride_k_seqlen = in_k.stride(1);
  const int64_t stride_k_nheads = in_k.stride(2);
  const int64_t stride_k_headdim = in_k.stride(3);
  const int64_t stride_cos_seqlen = cos.stride(0);
  const int64_t stride_cos_dim = cos.stride(1);
  
  const int64_t rotary_dim_half = rotary_dim / 2;
  const int64_t total_elements = batch_size * seqlen * nheads * rotary_dim_half;
  const int threads = 256;
  const int blocks = (total_elements + threads - 1) / threads;
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  VLLM_DISPATCH_FLOATING_TYPES(
      in_q.scalar_type(), "apply_vision_rotary_2c_kernel", [&] {
    vllm::apply_vision_rotary_2c_kernel<scalar_t>
        <<<blocks, threads, 0, stream>>>(
            out_q.data_ptr<scalar_t>(),
            out_k.data_ptr<scalar_t>(),
            in_q.data_ptr<scalar_t>(),
            in_k.data_ptr<scalar_t>(),
            cos.data_ptr<scalar_t>(),
            sin.data_ptr<scalar_t>(),
            batch_size,
            seqlen,
            nheads,
            headdim,
            rotary_dim,
            stride_q_batch,
            stride_q_seqlen,
            stride_q_nheads,
            stride_q_headdim,
            stride_k_batch,
            stride_k_seqlen,
            stride_k_nheads,
            stride_k_headdim,
            stride_cos_seqlen,
            stride_cos_dim);
  });
}

void apply_vision_rotary_2c_vec(
    torch::Tensor& out_q,           // [batch, seqlen, nheads, headdim]
    torch::Tensor& out_k,           // [batch, seqlen, nheads, headdim]
    const torch::Tensor& in_q,      // [batch, seqlen, nheads, headdim]
    const torch::Tensor& in_k,      // [batch, seqlen, nheads, headdim]
    const torch::Tensor& cos,       // [seqlen, rotary_dim / 2]
    const torch::Tensor& sin,       // [seqlen, rotary_dim / 2]
    int64_t rotary_dim) {

  const int64_t batch_size = in_q.size(0);
  const int64_t seqlen = in_q.size(1);
  const int64_t nheads = in_q.size(2);
  const int64_t headdim = in_q.size(3);

  TORCH_CHECK(in_q.dim() == 4, "in_q must be 4D tensor");
  TORCH_CHECK(in_k.dim() == 4, "in_k must be 4D tensor");
  TORCH_CHECK(cos.dim() == 2, "cos must be 2D tensor");
  TORCH_CHECK(sin.dim() == 2, "sin must be 2D tensor");
  TORCH_CHECK(in_q.sizes() == in_k.sizes(), "in_q and in_k must have same shape");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have same shape");
  TORCH_CHECK(cos.size(1) * 2 == rotary_dim, "cos second dim must be rotary_dim / 2");
  TORCH_CHECK(rotary_dim <= headdim, "rotary_dim must be <= headdim");
  TORCH_CHECK(cos.size(0) >= seqlen, "cos first dim must be >= seqlen");

  const int64_t stride_q_batch = in_q.stride(0);
  const int64_t stride_q_seqlen = in_q.stride(1);
  const int64_t stride_q_nheads = in_q.stride(2);
  const int64_t stride_q_headdim = in_q.stride(3);
  const int64_t stride_k_batch = in_k.stride(0);
  const int64_t stride_k_seqlen = in_k.stride(1);
  const int64_t stride_k_nheads = in_k.stride(2);
  const int64_t stride_k_headdim = in_k.stride(3);
  const int64_t stride_cos_seqlen = cos.stride(0);
  const int64_t stride_cos_dim = cos.stride(1);

  const int64_t rotary_dim_half = rotary_dim / 2;
  const int64_t total_elements = batch_size * seqlen * nheads * rotary_dim_half;
  const int threads = 256;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const bool contiguous_inputs = (stride_q_headdim == 1) && (stride_k_headdim == 1) &&
                         (stride_cos_dim == 1);
  const uintptr_t align_mask =
      reinterpret_cast<uintptr_t>(out_q.data_ptr()) |
      reinterpret_cast<uintptr_t>(out_k.data_ptr()) |
      reinterpret_cast<uintptr_t>(in_q.data_ptr()) |
      reinterpret_cast<uintptr_t>(in_k.data_ptr()) |
      reinterpret_cast<uintptr_t>(cos.data_ptr()) |
      reinterpret_cast<uintptr_t>(sin.data_ptr());
  const bool aligned_16b = (align_mask & 0xF) == 0;

  VLLM_DISPATCH_FLOATING_TYPES(
      in_q.scalar_type(), "apply_vision_rotary_2c_vec_kernel", [&] {
    constexpr int kVecWidth = 16 / sizeof(scalar_t);
    const bool can_vectorize = (rotary_dim_half % kVecWidth) == 0;

    if (contiguous_inputs && aligned_16b && can_vectorize) {
      const int64_t total_vec = total_elements / kVecWidth;
      const int vec_blocks = (total_vec + threads - 1) / threads;
      vllm::apply_vision_rotary_2c_vec_kernel<scalar_t>
          <<<vec_blocks, threads, 0, stream>>>(
              out_q.data_ptr<scalar_t>(),
              out_k.data_ptr<scalar_t>(),
              in_q.data_ptr<scalar_t>(),
              in_k.data_ptr<scalar_t>(),
              cos.data_ptr<scalar_t>(),
              sin.data_ptr<scalar_t>(),
              batch_size,
              seqlen,
              nheads,
              headdim,
              rotary_dim,
              stride_q_batch,
              stride_q_seqlen,
              stride_q_nheads,
              stride_q_headdim,
              stride_k_batch,
              stride_k_seqlen,
              stride_k_nheads,
              stride_k_headdim,
              stride_cos_seqlen,
              stride_cos_dim);
    } else {
      // Fallback to scalar path when vectorization constraints are not met.
      const int blocks = (total_elements + threads - 1) / threads;
      vllm::apply_vision_rotary_2c_kernel<scalar_t>
          <<<blocks, threads, 0, stream>>>(
              out_q.data_ptr<scalar_t>(),
              out_k.data_ptr<scalar_t>(),
              in_q.data_ptr<scalar_t>(),
              in_k.data_ptr<scalar_t>(),
              cos.data_ptr<scalar_t>(),
              sin.data_ptr<scalar_t>(),
              batch_size,
              seqlen,
              nheads,
              headdim,
              rotary_dim,
              stride_q_batch,
              stride_q_seqlen,
              stride_q_nheads,
              stride_q_headdim,
              stride_k_batch,
              stride_k_seqlen,
              stride_k_nheads,
              stride_k_headdim,
              stride_cos_seqlen,
              stride_cos_dim);
    }
  });
}