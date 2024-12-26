#include "core/registration.h"
#include "fbgemm_fp8_rowwise/ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, fp8gemm_ops) {
    // fp8 GEMM ops
    fp8gemm_ops.def("f8f8bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_) -> (Tensor)", &f8f8bf16_rowwise);
    fp8gemm_ops.impl("f8f8bf16_rowwise", torch::kCUDA, &f8f8bf16_rowwise);
    fp8gemm_ops.def("f8f8bf16_rowwise_out(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor !out, Tensor? bias, bool use_) -> ()", &f8f8bf16_rowwise_out);
    fp8gemm_ops.impl("f8f8bf16_rowwise_out", torch::kCUDA, &f8f8bf16_rowwise_out);

    // fp8gemm_ops.def("f8f8bf16_rowwise", &f8f8bf16_rowwise);
    // fp8gemm_ops.def("f8f8bf16_rowwise_out", &f8f8bf16_rowwise_out);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)