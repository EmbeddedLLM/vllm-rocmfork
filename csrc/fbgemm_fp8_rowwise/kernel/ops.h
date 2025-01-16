
#pragma once

#include <torch/all.h>


at::Tensor f8f8bf16_rowwise_32x32x16_11111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111611(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111611(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_113211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_113211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_121611(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121611(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_116111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_116211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_132111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_211611(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211611(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_221611(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221611(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_216111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_216211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_232111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_232111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42811(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_48111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_48211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_416111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_416111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82411(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_84111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_84211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_88111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_88111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_161111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_161211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_162111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_162211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162211(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_164111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_164111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_321111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_322111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_322111(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111612(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111612(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_116112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_211612(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211612(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_216112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41812(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_48112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81412(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_84112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_161112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_161212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161212(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_162112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_321112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321112(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11814(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11814(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21814(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21814(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41414(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81214(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_161114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161114(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11418(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11418(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21418(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21418(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41218(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81118(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_112116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_121116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_211116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_212116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_212116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_221116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_411116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_411116(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111621(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111621(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_121621(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121621(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_116121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_116221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_132121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22821(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_216121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42421(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_48121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82221(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_84121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_161121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_162121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162121(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11822(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11822(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111622(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111622(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12822(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12822(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_116122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21822(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21822(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41422(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81222(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_161122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161122(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11424(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11424(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11824(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11824(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12424(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12424(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21424(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21424(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41224(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81124(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11228(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11228(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11428(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11428(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12228(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12228(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21228(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21228(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11841(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11841(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12841(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12841(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_116141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22441(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_28141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42241(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_44141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_82141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82141(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11442(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11442(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11842(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11842(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12442(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12442(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21442(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21442(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41242(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_81142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81142(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11244(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11244(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11444(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11444(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12244(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12244(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21244(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21244(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41144(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11481(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11481(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12481(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12481(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_18181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22281(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_24181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_42181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42181(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11282(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11282(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11482(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11482(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12282(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12282(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_14182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21282(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21282(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_22182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_41182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41182(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_112161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_121161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_122161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_122161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_141161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_141161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_211161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_221161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221161(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


