
#pragma once

#define BENCHMARK_KERNELS_DEF_IMPL \ 
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11811_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11811_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11811_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11811_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11811_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11811_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11811_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11811_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11811_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11811_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11811_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11811_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11811_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11811_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11811_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11811_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11811_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11811_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11811_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11811_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111611", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111611", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111611_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111611_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111611_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111611_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111611_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111611_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111611_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111611_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111611_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111611_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111611_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111611_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111611_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111611_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111611_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111611_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111611_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111611_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111611_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111611_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_113211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_113211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_113211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_113211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_113211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_113211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_113211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_113211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_113211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_113211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_113211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_113211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_113211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_113211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_113211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_113211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_113211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_113211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_113211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_113211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_113211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_113211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_113211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_113211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_113211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_113211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_113211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_113211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12811_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12811_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12811_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12811_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12811_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12811_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12811_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12811_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12811_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12811_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12811_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12811_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12811_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12811_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12811_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12811_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12811_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12811_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12811_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12811_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_121611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_121611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_121611", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_121611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121611", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121611_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121611_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121611_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121611_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121611_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121611_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121611_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121611_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121611_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121611_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121611_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121611_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121611_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121611_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121611_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121611_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121611_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121611_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121611_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121611_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14811_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14811_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14811_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14811_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14811_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14811_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14811_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14811_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14811_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14811_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14811_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14811_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14811_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14811_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14811_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14811_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14811_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14811_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14811_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14811_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_132111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_132111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_132111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_132111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21811_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21811_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21811_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21811_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21811_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21811_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21811_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21811_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21811_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21811_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21811_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21811_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21811_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21811_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21811_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21811_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21811_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21811_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21811_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21811_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_211611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_211611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_211611", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_211611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211611", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211611_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211611_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211611_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211611_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211611_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211611_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211611_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211611_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211611_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211611_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211611_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211611_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211611_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211611_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211611_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211611_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211611_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211611_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211611_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211611_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22811_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22811_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22811_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22811_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22811_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22811_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22811_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22811_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22811_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22811_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22811_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22811_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22811_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22811_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22811_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22811_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22811_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22811_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22811_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22811_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_221611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_221611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_221611", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_221611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221611(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221611); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221611", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221611); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221611_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221611_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221611_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221611_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221611_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221611_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221611_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221611_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221611_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221611_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221611_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221611_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221611_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221611_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221611_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221611_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221611_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221611_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221611_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221611_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24811_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24811_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24811_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24811_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24811_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24811_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24811_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24811_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24811_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24811_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24811_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24811_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24811_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24811_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24811_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24811_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24811_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24811_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24811_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24811_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_216111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_216111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_216111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_216111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_216211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_216211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_216211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_216211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_232111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_232111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_232111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_232111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_232111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_232111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_232111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_232111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_232111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_232111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_232111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_232111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_232111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_232111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_232111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_232111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_232111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_232111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_232111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_232111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_232111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_232111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_232111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_232111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_232111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_232111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_232111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_232111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41811_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41811_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41811_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41811_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41811_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41811_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41811_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41811_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41811_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41811_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41811_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41811_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41811_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41811_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41811_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41811_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41811_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41811_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41811_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41811_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42811", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42811(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42811); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42811", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42811); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42811_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42811_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42811_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42811_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42811_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42811_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42811_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42811_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42811_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42811_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42811_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42811_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42811_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42811_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42811_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42811_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42811_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42811_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42811_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42811_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_48111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_48111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_48111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_48111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_48211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_48211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_48211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_48211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_416111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_416111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_416111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_416111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_416111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_416111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_416111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_416111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_416111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_416111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_416111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_416111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_416111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_416111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_416111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_416111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_416111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_416111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_416111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_416111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_416111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_416111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_416111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_416111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_416111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_416111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_416111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_416111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82411", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82411(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82411); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82411", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82411); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82411_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82411_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82411_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82411_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82411_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82411_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82411_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82411_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82411_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82411_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82411_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82411_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82411_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82411_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82411_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82411_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82411_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82411_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82411_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82411_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_84111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_84111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_84111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_84111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_84211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_84211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_84211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_84211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_88111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_88111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_88111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_88111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_88111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_88111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_88111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_88111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_88111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_88111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_88111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_88111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_88111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_88111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_88111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_88111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_88111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_88111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_88111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_88111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_88111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_88111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_88111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_88111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_88111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_88111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_88111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_88111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_162111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_162111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_162111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_162111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_162211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_162211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_162211", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_162211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162211(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162211); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162211", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162211); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162211_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162211_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162211_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162211_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162211_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162211_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162211_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162211_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162211_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162211_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162211_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162211_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162211_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162211_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162211_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162211_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162211_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162211_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162211_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162211_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_164111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_164111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_164111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_164111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_164111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_164111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_164111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_164111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_164111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_164111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_164111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_164111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_164111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_164111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_164111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_164111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_164111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_164111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_164111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_164111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_164111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_164111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_164111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_164111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_164111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_164111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_164111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_164111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_321111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_321111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_321111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_321111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_322111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_322111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_322111", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_322111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_322111(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_322111); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_322111", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_322111); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_322111_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_322111_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_322111_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_322111_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_322111_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_322111_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_322111_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_322111_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_322111_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_322111_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_322111_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_322111_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_322111_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_322111_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_322111_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_322111_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_322111_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_322111_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_322111_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_322111_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11812_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11812_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11812_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11812_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11812_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11812_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11812_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11812_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11812_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11812_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11812_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11812_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11812_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11812_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11812_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11812_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11812_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11812_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11812_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11812_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111612(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111612); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111612", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111612); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111612(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111612); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111612", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111612); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111612_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111612_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111612_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111612_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111612_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111612_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111612_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111612_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111612_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111612_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111612_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111612_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111612_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111612_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111612_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111612_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111612_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111612_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111612_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111612_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12812_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12812_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12812_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12812_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12812_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12812_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12812_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12812_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12812_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12812_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12812_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12812_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12812_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12812_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12812_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12812_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12812_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12812_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12812_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12812_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21812_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21812_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21812_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21812_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21812_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21812_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21812_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21812_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21812_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21812_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21812_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21812_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21812_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21812_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21812_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21812_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21812_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21812_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21812_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21812_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_211612(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_211612); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_211612", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_211612); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211612(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211612); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211612", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211612); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211612_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211612_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211612_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211612_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211612_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211612_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211612_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211612_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211612_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211612_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211612_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211612_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211612_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211612_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211612_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211612_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211612_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211612_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211612_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211612_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22812_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22812_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22812_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22812_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22812_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22812_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22812_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22812_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22812_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22812_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22812_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22812_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22812_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22812_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22812_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22812_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22812_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22812_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22812_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22812_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_216112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_216112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_216112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_216112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41812", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41812(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41812); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41812", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41812); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41812_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41812_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41812_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41812_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41812_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41812_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41812_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41812_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41812_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41812_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41812_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41812_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41812_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41812_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41812_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41812_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41812_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41812_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41812_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41812_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_48112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_48112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_48112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_48112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81412", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81412(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81412); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81412", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81412); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81412_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81412_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81412_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81412_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81412_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81412_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81412_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81412_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81412_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81412_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81412_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81412_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81412_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81412_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81412_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81412_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81412_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81412_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81412_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81412_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_84112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_84112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_84112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_84112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161212", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161212(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161212); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161212", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161212); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161212_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161212_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161212_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161212_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161212_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161212_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161212_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161212_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161212_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161212_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161212_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161212_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161212_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161212_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161212_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161212_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161212_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161212_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161212_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161212_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_162112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_162112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_162112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_162112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_321112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_321112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_321112", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_321112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321112(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321112); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321112", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321112); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321112_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321112_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321112_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321112_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321112_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321112_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321112_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321112_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321112_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321112_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321112_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321112_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321112_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321112_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321112_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321112_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_321112_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_321112_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_321112_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_321112_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11414_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11414_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11414_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11414_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11414_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11414_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11414_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11414_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11414_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11414_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11414_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11414_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11414_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11414_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11414_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11414_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11414_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11414_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11414_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11414_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11814(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11814); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11814", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11814); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11814(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11814); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11814", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11814); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11814_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11814_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11814_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11814_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11814_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11814_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11814_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11814_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11814_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11814_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11814_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11814_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11814_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11814_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11814_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11814_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11814_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11814_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11814_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11814_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12414_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12414_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12414_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12414_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12414_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12414_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12414_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12414_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12414_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12414_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12414_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12414_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12414_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12414_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12414_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12414_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12414_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12414_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12414_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12414_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21414_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21414_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21414_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21414_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21414_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21414_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21414_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21414_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21414_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21414_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21414_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21414_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21414_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21414_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21414_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21414_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21414_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21414_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21414_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21414_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21814(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21814); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21814", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21814); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21814(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21814); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21814", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21814); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21814_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21814_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21814_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21814_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21814_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21814_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21814_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21814_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21814_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21814_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21814_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21814_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21814_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21814_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21814_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21814_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21814_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21814_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21814_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21814_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22414_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22414_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22414_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22414_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22414_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22414_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22414_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22414_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22414_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22414_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22414_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22414_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22414_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22414_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22414_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22414_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22414_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22414_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22414_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22414_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41414", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41414(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41414); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41414", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41414); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41414_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41414_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41414_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41414_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41414_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41414_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41414_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41414_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41414_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41414_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41414_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41414_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41414_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41414_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41414_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41414_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41414_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41414_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41414_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41414_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81214", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81214(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81214); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81214", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81214); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81214_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81214_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81214_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81214_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81214_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81214_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81214_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81214_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81214_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81214_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81214_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81214_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81214_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81214_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81214_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81214_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81214_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81214_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81214_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81214_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161114", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161114(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161114); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161114", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161114); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161114_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161114_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161114_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161114_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161114_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161114_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161114_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161114_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161114_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161114_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161114_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161114_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161114_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161114_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161114_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161114_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161114_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161114_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161114_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161114_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11218_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11218_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11218_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11218_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11218_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11218_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11218_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11218_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11218_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11218_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11218_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11218_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11218_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11218_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11218_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11218_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11218_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11218_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11218_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11218_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11418(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11418); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11418", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11418); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11418(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11418); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11418", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11418); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11418_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11418_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11418_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11418_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11418_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11418_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11418_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11418_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11418_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11418_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11418_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11418_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11418_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11418_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11418_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11418_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11418_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11418_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11418_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11418_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12218_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12218_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12218_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12218_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12218_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12218_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12218_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12218_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12218_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12218_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12218_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12218_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12218_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12218_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12218_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12218_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12218_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12218_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12218_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12218_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21218_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21218_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21218_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21218_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21218_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21218_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21218_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21218_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21218_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21218_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21218_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21218_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21218_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21218_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21218_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21218_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21218_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21218_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21218_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21218_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21418(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21418); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21418", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21418); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21418(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21418); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21418", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21418); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21418_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21418_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21418_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21418_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21418_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21418_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21418_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21418_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21418_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21418_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21418_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21418_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21418_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21418_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21418_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21418_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21418_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21418_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21418_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21418_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22218_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22218_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22218_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22218_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22218_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22218_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22218_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22218_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22218_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22218_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22218_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22218_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22218_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22218_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22218_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22218_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22218_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22218_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22218_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22218_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41218", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41218(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41218); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41218", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41218); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41218_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41218_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41218_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41218_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41218_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41218_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41218_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41218_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41218_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41218_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41218_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41218_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41218_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41218_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41218_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41218_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41218_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41218_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41218_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41218_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81118", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81118(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81118); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81118", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81118); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81118_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81118_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81118_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81118_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81118_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81118_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81118_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81118_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81118_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81118_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81118_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81118_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81118_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81118_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81118_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81118_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81118_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81118_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81118_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81118_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111116_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111116_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111116_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111116_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111116_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111116_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111116_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111116_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111116_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111116_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111116_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111116_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111116_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111116_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111116_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111116_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111116_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111116_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111116_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111116_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_112116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_112116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_112116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_112116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112116_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112116_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112116_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112116_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112116_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112116_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112116_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112116_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112116_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112116_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112116_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112116_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112116_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112116_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112116_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112116_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112116_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112116_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112116_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112116_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_121116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_121116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_121116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_121116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121116_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121116_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121116_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121116_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121116_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121116_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121116_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121116_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121116_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121116_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121116_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121116_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121116_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121116_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121116_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121116_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121116_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121116_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121116_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121116_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_211116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_211116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_211116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_211116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211116_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211116_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211116_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211116_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211116_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211116_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211116_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211116_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211116_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211116_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211116_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211116_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211116_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211116_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211116_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211116_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211116_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211116_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211116_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211116_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_212116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_212116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_212116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_212116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_212116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_212116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_212116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_212116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_212116_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_212116_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_212116_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_212116_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_212116_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_212116_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_212116_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_212116_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_212116_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_212116_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_212116_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_212116_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_212116_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_212116_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_212116_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_212116_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_212116_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_212116_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_212116_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_212116_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_221116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_221116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_221116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_221116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221116_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221116_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221116_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221116_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221116_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221116_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221116_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221116_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221116_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221116_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221116_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221116_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221116_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221116_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221116_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221116_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221116_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221116_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221116_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221116_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_411116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_411116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_411116", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_411116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_411116(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_411116); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_411116", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_411116); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_411116_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_411116_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_411116_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_411116_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_411116_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_411116_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_411116_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_411116_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_411116_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_411116_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_411116_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_411116_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_411116_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_411116_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_411116_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_411116_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_411116_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_411116_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_411116_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_411116_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11821_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11821_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11821_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11821_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11821_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11821_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11821_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11821_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11821_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11821_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11821_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11821_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11821_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11821_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11821_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11821_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11821_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11821_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11821_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11821_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111621(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111621); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111621", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111621); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111621(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111621); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111621", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111621); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111621_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111621_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111621_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111621_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111621_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111621_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111621_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111621_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111621_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111621_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111621_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111621_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111621_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111621_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111621_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111621_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111621_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111621_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111621_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111621_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12821_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12821_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12821_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12821_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12821_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12821_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12821_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12821_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12821_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12821_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12821_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12821_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12821_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12821_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12821_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12821_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12821_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12821_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12821_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12821_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_121621(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_121621); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_121621", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_121621); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121621(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121621); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121621", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121621); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121621_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121621_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121621_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121621_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121621_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121621_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121621_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121621_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121621_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121621_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121621_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121621_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121621_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121621_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121621_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121621_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121621_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121621_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121621_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121621_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14821_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14821_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14821_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14821_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14821_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14821_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14821_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14821_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14821_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14821_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14821_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14821_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14821_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14821_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14821_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14821_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14821_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14821_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14821_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14821_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_132121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_132121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_132121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_132121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_132121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_132121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_132121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_132121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21821_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21821_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21821_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21821_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21821_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21821_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21821_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21821_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21821_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21821_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21821_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21821_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21821_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21821_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21821_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21821_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21821_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21821_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21821_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21821_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22821", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22821(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22821); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22821", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22821); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22821_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22821_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22821_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22821_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22821_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22821_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22821_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22821_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22821_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22821_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22821_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22821_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22821_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22821_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22821_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22821_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22821_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22821_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22821_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22821_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_216121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_216121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_216121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_216121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_216121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_216121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_216121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_216121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42421", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42421(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42421); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42421", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42421); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42421_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42421_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42421_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42421_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42421_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42421_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42421_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42421_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42421_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42421_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42421_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42421_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42421_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42421_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42421_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42421_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42421_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42421_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42421_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42421_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_48121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_48121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_48121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_48121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_48121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_48121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_48121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_48121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82221", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82221(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82221); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82221", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82221); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82221_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82221_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82221_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82221_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82221_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82221_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82221_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82221_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82221_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82221_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82221_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82221_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82221_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82221_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82221_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82221_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82221_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82221_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82221_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82221_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_84121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_84121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_84121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_84121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_84121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_84121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_84121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_84121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_162121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_162121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_162121", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_162121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162121(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162121); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162121", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162121); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162121_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162121_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162121_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162121_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162121_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162121_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162121_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162121_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162121_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162121_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162121_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162121_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162121_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162121_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162121_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162121_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_162121_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_162121_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_162121_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_162121_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11422_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11422_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11422_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11422_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11422_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11422_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11422_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11422_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11422_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11422_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11422_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11422_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11422_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11422_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11422_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11422_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11422_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11422_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11422_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11422_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11822", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11822", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11822_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11822_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11822_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11822_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11822_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11822_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11822_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11822_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11822_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11822_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11822_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11822_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11822_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11822_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11822_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11822_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11822_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11822_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11822_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11822_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111622(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111622); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111622", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111622); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111622(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111622); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111622", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111622); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111622_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111622_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111622_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111622_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111622_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111622_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111622_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111622_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111622_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111622_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111622_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111622_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111622_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111622_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111622_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111622_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111622_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111622_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111622_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111622_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12422_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12422_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12422_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12422_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12422_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12422_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12422_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12422_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12422_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12422_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12422_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12422_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12422_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12422_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12422_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12422_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12422_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12422_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12422_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12422_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12822", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12822", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12822_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12822_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12822_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12822_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12822_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12822_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12822_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12822_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12822_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12822_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12822_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12822_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12822_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12822_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12822_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12822_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12822_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12822_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12822_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12822_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14422_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14422_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14422_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14422_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14422_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14422_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14422_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14422_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14422_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14422_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14422_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14422_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14422_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14422_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14422_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14422_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14422_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14422_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14422_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14422_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21422_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21422_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21422_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21422_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21422_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21422_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21422_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21422_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21422_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21422_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21422_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21422_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21422_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21422_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21422_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21422_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21422_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21422_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21422_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21422_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21822", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21822(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21822); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21822", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21822); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21822_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21822_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21822_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21822_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21822_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21822_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21822_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21822_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21822_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21822_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21822_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21822_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21822_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21822_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21822_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21822_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21822_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21822_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21822_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21822_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22422_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22422_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22422_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22422_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22422_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22422_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22422_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22422_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22422_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22422_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22422_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22422_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22422_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22422_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22422_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22422_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22422_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22422_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22422_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22422_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41422", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41422(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41422); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41422", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41422); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41422_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41422_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41422_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41422_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41422_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41422_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41422_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41422_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41422_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41422_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41422_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41422_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41422_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41422_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41422_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41422_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41422_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41422_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41422_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41422_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81222", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81222(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81222); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81222", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81222); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81222_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81222_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81222_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81222_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81222_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81222_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81222_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81222_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81222_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81222_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81222_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81222_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81222_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81222_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81222_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81222_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81222_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81222_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81222_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81222_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_161122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_161122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_161122", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_161122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161122(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161122); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161122", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161122); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161122_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161122_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161122_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161122_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161122_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161122_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161122_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161122_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161122_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161122_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161122_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161122_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161122_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161122_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161122_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161122_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_161122_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_161122_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_161122_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_161122_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11224_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11224_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11224_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11224_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11224_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11224_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11224_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11224_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11224_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11224_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11224_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11224_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11224_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11224_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11224_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11224_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11224_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11224_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11224_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11224_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11424", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11424", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11424_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11424_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11424_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11424_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11424_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11424_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11424_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11424_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11424_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11424_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11424_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11424_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11424_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11424_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11424_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11424_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11424_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11424_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11424_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11424_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11824(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11824); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11824", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11824); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11824(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11824); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11824", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11824); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11824_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11824_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11824_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11824_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11824_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11824_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11824_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11824_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11824_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11824_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11824_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11824_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11824_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11824_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11824_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11824_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11824_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11824_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11824_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11824_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12224_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12224_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12224_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12224_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12224_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12224_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12224_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12224_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12224_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12224_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12224_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12224_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12224_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12224_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12224_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12224_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12224_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12224_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12224_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12224_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12424", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12424", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12424_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12424_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12424_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12424_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12424_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12424_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12424_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12424_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12424_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12424_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12424_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12424_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12424_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12424_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12424_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12424_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12424_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12424_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12424_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12424_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14224_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14224_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14224_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14224_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14224_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14224_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14224_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14224_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14224_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14224_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14224_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14224_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14224_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14224_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14224_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14224_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14224_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14224_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14224_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14224_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21224_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21224_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21224_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21224_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21224_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21224_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21224_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21224_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21224_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21224_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21224_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21224_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21224_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21224_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21224_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21224_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21224_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21224_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21224_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21224_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21424", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21424(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21424); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21424", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21424); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21424_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21424_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21424_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21424_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21424_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21424_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21424_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21424_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21424_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21424_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21424_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21424_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21424_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21424_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21424_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21424_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21424_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21424_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21424_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21424_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22224_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22224_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22224_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22224_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22224_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22224_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22224_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22224_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22224_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22224_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22224_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22224_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22224_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22224_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22224_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22224_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22224_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22224_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22224_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22224_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41224", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41224(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41224); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41224", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41224); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41224_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41224_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41224_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41224_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41224_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41224_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41224_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41224_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41224_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41224_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41224_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41224_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41224_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41224_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41224_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41224_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41224_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41224_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41224_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41224_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81124", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81124(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81124); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81124", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81124); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81124_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81124_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81124_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81124_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81124_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81124_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81124_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81124_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81124_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81124_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81124_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81124_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81124_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81124_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81124_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81124_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81124_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81124_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81124_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81124_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11128_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11128_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11128_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11128_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11128_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11128_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11128_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11128_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11128_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11128_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11128_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11128_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11128_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11128_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11128_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11128_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11128_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11128_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11128_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11128_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11228", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11228", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11228_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11228_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11228_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11228_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11228_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11228_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11228_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11228_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11228_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11228_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11228_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11228_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11228_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11228_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11228_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11228_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11228_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11228_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11228_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11228_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11428(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11428); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11428", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11428); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11428(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11428); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11428", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11428); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11428_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11428_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11428_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11428_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11428_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11428_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11428_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11428_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11428_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11428_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11428_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11428_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11428_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11428_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11428_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11428_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11428_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11428_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11428_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11428_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12128_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12128_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12128_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12128_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12128_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12128_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12128_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12128_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12128_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12128_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12128_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12128_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12128_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12128_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12128_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12128_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12128_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12128_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12128_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12128_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12228", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12228", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12228_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12228_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12228_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12228_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12228_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12228_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12228_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12228_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12228_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12228_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12228_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12228_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12228_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12228_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12228_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12228_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12228_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12228_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12228_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12228_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14128_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14128_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14128_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14128_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14128_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14128_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14128_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14128_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14128_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14128_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14128_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14128_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14128_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14128_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14128_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14128_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14128_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14128_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14128_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14128_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21128_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21128_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21128_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21128_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21128_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21128_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21128_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21128_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21128_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21128_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21128_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21128_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21128_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21128_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21128_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21128_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21128_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21128_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21128_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21128_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21228", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21228(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21228); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21228", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21228); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21228_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21228_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21228_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21228_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21228_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21228_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21228_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21228_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21228_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21228_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21228_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21228_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21228_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21228_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21228_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21228_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21228_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21228_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21228_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21228_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22128_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22128_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22128_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22128_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22128_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22128_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22128_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22128_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22128_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22128_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22128_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22128_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22128_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22128_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22128_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22128_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22128_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22128_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22128_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22128_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41128", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41128_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41128_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41128_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41128_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41128_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41128_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41128_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41128_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41128_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41128_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41128_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41128_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41128_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41128_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41128_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41128_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41128_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41128_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41128_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41128_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11441_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11441_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11441_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11441_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11441_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11441_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11441_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11441_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11441_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11441_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11441_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11441_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11441_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11441_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11441_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11441_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11441_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11441_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11441_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11441_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11841(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11841); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11841", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11841); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11841(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11841); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11841", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11841); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11841_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11841_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11841_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11841_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11841_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11841_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11841_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11841_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11841_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11841_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11841_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11841_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11841_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11841_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11841_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11841_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11841_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11841_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11841_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11841_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12441_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12441_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12441_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12441_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12441_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12441_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12441_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12441_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12441_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12441_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12441_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12441_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12441_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12441_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12441_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12441_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12441_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12441_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12441_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12441_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12841(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12841); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12841", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12841); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12841(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12841); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12841", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12841); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12841_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12841_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12841_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12841_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12841_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12841_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12841_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12841_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12841_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12841_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12841_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12841_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12841_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12841_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12841_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12841_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12841_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12841_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12841_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12841_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14441_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14441_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14441_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14441_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14441_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14441_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14441_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14441_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14441_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14441_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14441_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14441_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14441_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14441_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14441_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14441_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14441_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14441_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14441_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14441_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_116141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_116141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_116141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_116141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_116141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_116141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_116141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_116141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21441_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21441_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21441_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21441_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21441_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21441_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21441_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21441_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21441_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21441_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21441_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21441_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21441_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21441_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21441_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21441_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21441_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21441_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21441_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21441_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22441", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22441(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22441); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22441", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22441); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22441_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22441_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22441_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22441_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22441_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22441_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22441_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22441_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22441_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22441_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22441_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22441_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22441_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22441_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22441_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22441_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22441_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22441_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22441_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22441_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_28141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_28141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_28141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_28141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_28141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_28141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_28141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_28141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42241", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42241(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42241); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42241", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42241); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42241_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42241_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42241_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42241_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42241_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42241_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42241_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42241_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42241_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42241_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42241_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42241_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42241_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42241_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42241_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42241_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42241_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42241_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42241_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42241_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_44141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_44141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_44141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_44141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_44141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_44141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_44141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_44141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_82141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_82141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_82141", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_82141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82141(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82141); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82141", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82141); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82141_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82141_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82141_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82141_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82141_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82141_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82141_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82141_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82141_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82141_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82141_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82141_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82141_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82141_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82141_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82141_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_82141_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_82141_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_82141_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_82141_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11242_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11242_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11242_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11242_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11242_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11242_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11242_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11242_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11242_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11242_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11242_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11242_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11242_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11242_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11242_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11242_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11242_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11242_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11242_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11242_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11442", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11442", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11442_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11442_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11442_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11442_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11442_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11442_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11442_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11442_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11442_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11442_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11442_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11442_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11442_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11442_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11442_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11442_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11442_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11442_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11442_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11442_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11842(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11842); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11842", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11842); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11842(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11842); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11842", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11842); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11842_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11842_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11842_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11842_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11842_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11842_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11842_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11842_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11842_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11842_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11842_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11842_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11842_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11842_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11842_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11842_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11842_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11842_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11842_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11842_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12242_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12242_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12242_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12242_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12242_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12242_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12242_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12242_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12242_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12242_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12242_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12242_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12242_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12242_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12242_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12242_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12242_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12242_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12242_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12242_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12442", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12442", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12442_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12442_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12442_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12442_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12442_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12442_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12442_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12442_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12442_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12442_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12442_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12442_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12442_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12442_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12442_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12442_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12442_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12442_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12442_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12442_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14242_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14242_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14242_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14242_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14242_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14242_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14242_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14242_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14242_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14242_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14242_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14242_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14242_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14242_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14242_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14242_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14242_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14242_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14242_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14242_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21242_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21242_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21242_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21242_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21242_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21242_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21242_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21242_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21242_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21242_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21242_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21242_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21242_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21242_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21242_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21242_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21242_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21242_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21242_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21242_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21442", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21442(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21442); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21442", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21442); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21442_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21442_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21442_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21442_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21442_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21442_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21442_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21442_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21442_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21442_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21442_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21442_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21442_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21442_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21442_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21442_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21442_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21442_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21442_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21442_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22242_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22242_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22242_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22242_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22242_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22242_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22242_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22242_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22242_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22242_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22242_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22242_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22242_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22242_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22242_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22242_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22242_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22242_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22242_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22242_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41242", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41242(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41242); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41242", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41242); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41242_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41242_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41242_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41242_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41242_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41242_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41242_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41242_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41242_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41242_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41242_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41242_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41242_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41242_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41242_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41242_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41242_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41242_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41242_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41242_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_81142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_81142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_81142", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_81142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81142(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81142); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81142", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81142); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81142_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81142_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81142_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81142_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81142_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81142_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81142_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81142_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81142_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81142_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81142_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81142_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81142_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81142_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81142_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81142_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_81142_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_81142_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_81142_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_81142_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11144_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11144_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11144_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11144_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11144_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11144_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11144_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11144_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11144_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11144_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11144_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11144_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11144_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11144_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11144_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11144_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11144_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11144_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11144_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11144_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11244", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11244", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11244_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11244_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11244_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11244_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11244_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11244_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11244_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11244_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11244_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11244_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11244_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11244_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11244_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11244_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11244_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11244_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11244_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11244_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11244_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11244_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11444(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11444); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11444", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11444); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11444(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11444); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11444", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11444); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11444_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11444_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11444_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11444_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11444_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11444_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11444_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11444_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11444_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11444_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11444_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11444_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11444_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11444_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11444_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11444_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11444_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11444_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11444_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11444_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12144_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12144_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12144_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12144_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12144_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12144_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12144_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12144_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12144_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12144_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12144_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12144_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12144_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12144_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12144_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12144_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12144_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12144_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12144_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12144_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12244", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12244", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12244_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12244_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12244_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12244_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12244_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12244_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12244_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12244_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12244_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12244_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12244_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12244_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12244_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12244_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12244_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12244_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12244_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12244_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12244_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12244_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14144_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14144_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14144_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14144_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14144_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14144_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14144_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14144_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14144_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14144_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14144_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14144_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14144_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14144_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14144_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14144_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14144_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14144_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14144_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14144_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21144_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21144_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21144_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21144_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21144_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21144_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21144_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21144_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21144_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21144_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21144_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21144_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21144_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21144_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21144_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21144_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21144_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21144_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21144_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21144_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21244", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21244(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21244); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21244", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21244); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21244_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21244_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21244_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21244_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21244_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21244_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21244_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21244_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21244_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21244_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21244_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21244_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21244_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21244_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21244_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21244_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21244_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21244_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21244_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21244_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22144_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22144_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22144_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22144_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22144_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22144_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22144_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22144_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22144_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22144_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22144_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22144_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22144_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22144_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22144_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22144_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22144_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22144_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22144_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22144_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41144", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41144(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41144); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41144", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41144); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41144_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41144_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41144_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41144_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41144_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41144_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41144_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41144_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41144_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41144_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41144_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41144_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41144_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41144_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41144_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41144_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41144_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41144_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41144_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41144_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11281_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11281_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11281_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11281_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11281_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11281_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11281_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11281_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11281_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11281_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11281_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11281_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11281_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11281_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11281_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11281_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11281_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11281_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11281_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11281_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11481(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11481); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11481", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11481); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11481(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11481); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11481", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11481); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11481_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11481_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11481_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11481_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11481_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11481_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11481_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11481_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11481_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11481_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11481_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11481_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11481_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11481_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11481_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11481_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11481_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11481_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11481_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11481_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12281_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12281_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12281_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12281_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12281_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12281_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12281_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12281_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12281_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12281_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12281_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12281_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12281_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12281_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12281_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12281_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12281_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12281_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12281_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12281_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12481(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12481); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12481", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12481); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12481(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12481); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12481", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12481); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12481_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12481_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12481_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12481_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12481_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12481_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12481_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12481_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12481_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12481_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12481_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12481_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12481_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12481_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12481_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12481_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12481_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12481_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12481_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12481_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14281_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14281_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14281_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14281_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14281_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14281_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14281_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14281_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14281_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14281_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14281_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14281_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14281_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14281_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14281_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14281_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14281_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14281_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14281_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14281_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_18181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_18181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_18181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_18181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_18181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_18181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_18181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_18181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21281_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21281_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21281_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21281_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21281_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21281_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21281_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21281_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21281_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21281_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21281_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21281_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21281_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21281_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21281_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21281_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21281_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21281_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21281_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21281_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22281", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22281(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22281); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22281", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22281); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22281_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22281_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22281_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22281_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22281_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22281_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22281_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22281_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22281_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22281_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22281_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22281_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22281_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22281_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22281_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22281_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22281_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22281_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22281_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22281_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_24181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_24181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_24181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_24181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_24181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_24181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_24181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_24181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_42181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_42181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_42181", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_42181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42181(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42181); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42181", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42181); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42181_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42181_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42181_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42181_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42181_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42181_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42181_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42181_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42181_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42181_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42181_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42181_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42181_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42181_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42181_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42181_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_42181_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_42181_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_42181_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_42181_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11182_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11182_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11182_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11182_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11182_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11182_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11182_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11182_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11182_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11182_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11182_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11182_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11182_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11182_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11182_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11182_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11182_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11182_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11182_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11182_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11282", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11282", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11282_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11282_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11282_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11282_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11282_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11282_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11282_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11282_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11282_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11282_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11282_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11282_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11282_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11282_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11282_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11282_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11282_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11282_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11282_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11282_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_11482(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_11482); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_11482", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_11482); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11482(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11482); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11482", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11482); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11482_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11482_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11482_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11482_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11482_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11482_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11482_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11482_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11482_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11482_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11482_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11482_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11482_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11482_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11482_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11482_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_11482_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_11482_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_11482_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_11482_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12182_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12182_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12182_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12182_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12182_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12182_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12182_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12182_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12182_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12182_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12182_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12182_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12182_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12182_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12182_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12182_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12182_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12182_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12182_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12182_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_12282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_12282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_12282", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_12282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12282", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12282_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12282_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12282_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12282_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12282_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12282_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12282_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12282_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12282_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12282_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12282_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12282_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12282_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12282_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12282_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12282_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_12282_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_12282_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_12282_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_12282_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_14182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_14182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_14182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_14182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14182_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14182_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14182_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14182_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14182_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14182_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14182_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14182_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14182_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14182_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14182_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14182_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14182_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14182_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14182_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14182_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_14182_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_14182_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_14182_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_14182_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21182_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21182_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21182_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21182_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21182_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21182_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21182_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21182_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21182_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21182_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21182_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21182_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21182_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21182_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21182_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21182_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21182_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21182_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21182_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21182_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_21282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_21282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_21282", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_21282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21282(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21282); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21282", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21282); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21282_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21282_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21282_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21282_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21282_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21282_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21282_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21282_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21282_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21282_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21282_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21282_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21282_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21282_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21282_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21282_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_21282_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_21282_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_21282_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_21282_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_22182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_22182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_22182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_22182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22182_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22182_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22182_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22182_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22182_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22182_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22182_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22182_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22182_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22182_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22182_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22182_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22182_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22182_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22182_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22182_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_22182_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_22182_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_22182_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_22182_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_41182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_41182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_41182", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_41182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41182(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41182); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41182", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41182); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41182_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41182_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41182_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41182_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41182_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41182_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41182_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41182_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41182_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41182_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41182_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41182_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41182_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41182_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41182_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41182_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_41182_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_41182_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_41182_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_41182_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_111161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_111161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_111161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_111161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111161_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111161_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111161_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111161_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111161_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111161_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111161_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111161_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111161_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111161_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111161_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111161_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111161_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111161_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111161_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111161_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_111161_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_111161_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_111161_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_111161_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_112161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_112161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_112161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_112161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112161_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112161_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112161_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112161_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112161_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112161_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112161_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112161_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112161_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112161_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112161_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112161_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112161_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112161_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112161_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112161_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_112161_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_112161_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_112161_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_112161_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_121161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_121161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_121161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_121161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121161_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121161_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121161_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121161_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121161_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121161_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121161_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121161_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121161_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121161_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121161_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121161_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121161_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121161_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121161_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121161_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_121161_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_121161_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_121161_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_121161_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_122161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_122161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_122161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_122161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_122161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_122161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_122161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_122161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_122161_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_122161_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_122161_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_122161_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_122161_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_122161_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_122161_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_122161_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_122161_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_122161_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_122161_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_122161_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_122161_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_122161_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_122161_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_122161_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_122161_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_122161_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_122161_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_122161_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_141161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_141161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_141161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_141161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_141161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_141161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_141161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_141161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_141161_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_141161_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_141161_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_141161_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_141161_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_141161_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_141161_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_141161_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_141161_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_141161_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_141161_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_141161_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_141161_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_141161_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_141161_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_141161_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_141161_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_141161_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_141161_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_141161_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_211161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_211161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_211161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_211161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211161_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211161_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211161_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211161_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211161_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211161_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211161_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211161_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211161_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211161_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211161_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211161_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211161_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211161_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211161_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211161_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_211161_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_211161_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_211161_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_211161_sk2048); \
\
    fp8gemm_ops.def("f8f8bf16_rowwise_32x32x16_221161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_32x32x16_221161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_32x32x16_221161", torch::kCUDA, &f8f8bf16_rowwise_32x32x16_221161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221161(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221161); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221161", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221161); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221161_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221161_sk128); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221161_sk128", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221161_sk128); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221161_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221161_sk256); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221161_sk256", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221161_sk256); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221161_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221161_sk512); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221161_sk512", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221161_sk512); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221161_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221161_sk1024); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221161_sk1024", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221161_sk1024); \
    fp8gemm_ops.def("f8f8bf16_rowwise_16x16x32_221161_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &f8f8bf16_rowwise_16x16x32_221161_sk2048); \
    fp8gemm_ops.impl("f8f8bf16_rowwise_16x16x32_221161_sk2048", torch::kCUDA, &f8f8bf16_rowwise_16x16x32_221161_sk2048); \
\
