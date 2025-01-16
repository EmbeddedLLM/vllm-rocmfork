
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11811); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111611", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111611", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111611); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_113211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_113211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_113211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_113211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_113211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_113211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_113211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_113211); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12811); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_121611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_121611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_121611", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_121611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121611", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121611); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14811); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_132111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_132111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_132111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_132111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132111); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21811); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_211611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_211611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_211611", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_211611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211611", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211611); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22811); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_221611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_221611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_221611", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_221611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221611", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221611); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24811); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_216111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_216111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_216111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_216111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_216211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_216211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_216211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_216211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_232111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_232111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_232111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_232111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_232111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_232111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_232111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_232111); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41811); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42811); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_48111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_48111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_48111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_48111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_48211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_48211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_48211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_48211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_416111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_416111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_416111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_416111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_416111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_416111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_416111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_416111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82411); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_84111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_84111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_84111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_84111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_84211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_84211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_84211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_84211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_88111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_88111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_88111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_88111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_88111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_88111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_88111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_88111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_162111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_162111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_162111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_162111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_162211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_162211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_162211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_162211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162211); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_164111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_164111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_164111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_164111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_164111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_164111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_164111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_164111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_321111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_321111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_321111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_321111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321111); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_322111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_322111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_322111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_322111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_322111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_322111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_322111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_322111); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11412); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11812); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111612(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111612); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111612", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111612); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111612(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111612); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111612", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111612); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12412); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12812); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14412); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116112); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21412); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21812); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_211612(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_211612); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_211612", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_211612); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211612(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211612); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211612", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211612); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22412); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22812); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24412); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_216112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_216112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_216112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_216112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216112); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41412); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41812); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42412); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_48112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_48112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_48112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_48112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81412); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_84112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_84112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_84112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_84112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161212); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_162112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_162112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_162112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_162112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162112); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_321112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_321112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_321112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_321112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321112); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11414); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11814(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11814); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11814", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11814); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11814(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11814); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11814", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11814); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12414); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18114); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21414); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21814(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21814); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21814", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21814); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21814(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21814); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21814", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21814); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22414); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28114); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41414); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81214); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82114); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161114); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11418(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11418); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11418", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11418); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11418(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11418); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11418", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11418); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21418(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21418); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21418", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21418); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21418(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21418); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21418", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21418); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81118); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111116); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_112116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_112116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_112116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_112116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112116); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_121116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_121116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_121116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_121116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121116); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_211116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_211116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_211116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_211116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211116); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_212116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_212116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_212116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_212116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_212116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_212116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_212116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_212116); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_221116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_221116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_221116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_221116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221116); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_411116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_411116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_411116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_411116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_411116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_411116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_411116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_411116); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11421); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11821); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111621(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111621); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111621", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111621); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111621(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111621); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111621", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111621); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12421); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12821); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_121621(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_121621); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_121621", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_121621); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121621(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121621); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121621", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121621); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14421); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14821); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18421); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_132121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_132121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_132121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_132121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132121); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21421); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21821); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22421); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22821); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24421); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_216121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_216121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_216121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_216121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216121); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41421); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42421); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_48121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_48121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_48121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_48121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82221); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_84121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_84121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_84121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_84121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161121); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_162121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_162121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_162121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_162121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162121); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11422); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11822", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11822", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11822); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111622(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111622); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111622", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111622); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111622(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111622); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111622", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111622); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12422); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12822", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12822", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12822); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14422); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116122); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21422); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21822", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21822", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21822); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22422); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28122); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41422); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81222); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82122); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161122); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11424", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11424", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11424); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11824(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11824); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11824", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11824); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11824(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11824); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11824", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11824); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12424", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12424", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12424); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18124); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21424", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21424", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21424); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81124); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11428(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11428); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11428", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11428); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11428(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11428); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11428", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11428); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11441); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11841(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11841); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11841", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11841); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11841(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11841); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11841", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11841); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12441); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12841(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12841); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12841", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12841); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12841(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12841); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12841", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12841); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14441); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18241); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116141); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21441); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22441); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28141); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81141); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82141); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11442", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11442", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11442); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11842(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11842); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11842", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11842); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11842(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11842); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11842", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11842); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12442", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12442", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12442); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18142); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21442", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21442", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21442); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81142); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11444(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11444); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11444", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11444); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11444(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11444); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11444", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11444); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11481(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11481); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11481", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11481); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11481(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11481); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11481", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11481); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12481(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12481); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12481", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12481); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12481(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12481); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12481", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12481); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18181); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11482(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11482); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11482", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11482); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11482(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11482); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11482", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11482); \
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
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111161); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_112161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_112161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_112161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_112161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112161); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_121161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_121161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_121161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_121161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121161); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_122161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_122161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_122161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_122161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_122161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_122161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_122161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_122161); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_141161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_141161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_141161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_141161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_141161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_141161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_141161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_141161); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_211161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_211161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_211161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_211161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211161); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_221161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_221161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_221161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_221161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221161); \
\
