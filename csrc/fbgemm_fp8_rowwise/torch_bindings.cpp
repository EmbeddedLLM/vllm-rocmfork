#include "core/registration.h"
#include "fbgemm_fp8_rowwise/ops.h"

#include "kernel/benchmark_bindings.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, fp8gemm_ops) {
    // fp8 GEMM ops
    fp8gemm_ops.def("f8f8bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_instr1);
    fp8gemm_ops.impl("f8f8bf16_rowwise_instr1", torch::kCUDA, &f8f8bf16_rowwise_instr1);
    fp8gemm_ops.def("f8f8bf16_rowwise_instr1(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise);
    fp8gemm_ops.impl("f8f8bf16_rowwise", torch::kCUDA, &f8f8bf16_rowwise);
    fp8gemm_ops.def("f8f8bf16_rowwise_instr2(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_instr2);
    fp8gemm_ops.impl("f8f8bf16_rowwise_instr2", torch::kCUDA, &f8f8bf16_rowwise_instr2);
    
    fp8gemm_ops.def("f8f8bf16_rowwise_instr2_sk(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_instr2_sk);
    fp8gemm_ops.impl("f8f8bf16_rowwise_instr2_sk", torch::kCUDA, &f8f8bf16_rowwise_instr2_sk);

    // benchmarking ops
    BENCHMARK_KERNELS_DEF_IMPL
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)