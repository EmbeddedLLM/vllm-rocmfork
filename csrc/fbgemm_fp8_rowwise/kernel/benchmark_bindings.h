
#pragma once

#define BENCHMARK_KERNELS_DEF_IMPL \ 
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11218); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12218); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21218); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22218); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41218); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11124); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11224); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12124); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12224); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14124); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14224); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21124); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21224); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22124); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22224); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24124); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41124); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41224); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42124); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11128); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11228", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11228", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11228); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12128); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12228", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12228", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12228); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14128); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21128); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21228", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21228", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21228); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22128); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41128); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11142); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11242); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12142); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12242); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14142); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14242); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21142); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21242); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22142); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22242); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24142); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41142); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41242); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42142); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11144); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11244", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11244", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11244); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12144); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12244", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12244", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12244); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14144); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21144); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21244", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21244", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21244); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22144); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41144); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11148(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11148); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11148", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11148); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11148(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11148); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11148", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11148); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11248(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11248); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11248", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11248); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11248(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11248); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11248", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11248); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12148(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12148); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12148", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12148); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12148(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12148); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12148", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12148); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21148(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21148); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21148", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21148); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21148(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21148); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21148", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21148); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11181); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11281); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12181); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12281); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14181); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14281); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21181); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21281); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22181); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22281); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24181); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41181); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42181); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11182); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11282", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11282", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11282); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12182); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12282", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12282", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12282); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14182); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21182); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21282", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21282", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21282); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22182); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41182); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11184(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11184); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11184", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11184); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11184(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11184); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11184", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11184); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11284(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11284); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11284", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11284); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11284(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11284); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11284", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11284); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12184(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12184); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12184", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12184); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12184(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12184); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12184", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12184); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21184(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21184); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21184", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21184); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21184(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21184); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21184", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21184); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11188(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11188); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11188", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11188); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11188(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11188); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11188", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11188); \
\
