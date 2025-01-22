
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

at::Tensor f8f8bf16_rowwise_16x16x32_11111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11811_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11811_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11811_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11811_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11811_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_111611_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111611_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111611_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111611_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111611_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_113211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_113211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_113211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_113211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_113211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12811_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12811_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12811_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12811_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12811_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_121611_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121611_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121611_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121611_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121611_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14811_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14811_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14811_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14811_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14811_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_116111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_116211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_132111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21811_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21811_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21811_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21811_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21811_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_211611_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211611_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211611_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211611_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211611_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22811_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22811_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22811_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22811_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22811_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_221611_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221611_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221611_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221611_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221611_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24811_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24811_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24811_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24811_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24811_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_216111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_216211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_232111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_232111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_232111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_232111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_232111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41811_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41811_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41811_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41811_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41811_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42811_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42811_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42811_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42811_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42811_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_48111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_48211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_416111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_416111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_416111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_416111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_416111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82411_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82411_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82411_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82411_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82411_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_84111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_84211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_88111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_88111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_88111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_88111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_88111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_161111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_161211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_162111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_162211_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162211_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162211_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162211_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162211_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_164111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_164111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_164111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_164111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_164111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_321111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_322111_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_322111_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_322111_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_322111_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_322111_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11812_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11812_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11812_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11812_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11812_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_111612_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111612_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111612_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111612_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111612_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12812_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12812_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12812_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12812_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12812_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_116112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21812_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21812_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21812_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21812_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21812_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_211612_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211612_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211612_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211612_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211612_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22812_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22812_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22812_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22812_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22812_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_216112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41812_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41812_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41812_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41812_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41812_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_48112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81412_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81412_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81412_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81412_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81412_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_84112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_161112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_161212_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161212_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161212_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161212_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161212_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_162112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_321112_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321112_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321112_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321112_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_321112_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11414_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11414_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11414_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11414_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11414_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11814_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11814_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11814_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11814_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11814_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12414_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12414_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12414_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12414_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12414_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21414_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21414_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21414_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21414_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21414_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21814_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21814_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21814_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21814_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21814_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22414_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22414_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22414_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22414_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22414_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41414_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41414_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41414_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41414_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41414_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81214_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81214_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81214_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81214_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81214_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_161114_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161114_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161114_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161114_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161114_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11218_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11218_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11218_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11218_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11218_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11418_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11418_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11418_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11418_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11418_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12218_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12218_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12218_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12218_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12218_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21218_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21218_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21218_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21218_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21218_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21418_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21418_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21418_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21418_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21418_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22218_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22218_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22218_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22218_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22218_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41218_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41218_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41218_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41218_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41218_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81118_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81118_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81118_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81118_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81118_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_111116_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111116_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111116_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111116_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111116_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_112116_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112116_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112116_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112116_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112116_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_121116_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121116_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121116_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121116_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121116_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_211116_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211116_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211116_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211116_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211116_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_212116_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_212116_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_212116_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_212116_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_212116_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_221116_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221116_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221116_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221116_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221116_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_411116_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_411116_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_411116_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_411116_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_411116_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11821_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11821_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11821_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11821_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11821_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_111621_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111621_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111621_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111621_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111621_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12821_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12821_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12821_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12821_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12821_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_121621_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121621_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121621_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121621_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121621_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14821_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14821_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14821_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14821_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14821_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_116121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_116221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_132121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_132121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21821_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21821_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21821_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21821_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21821_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22821_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22821_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22821_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22821_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22821_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_216121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_216121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42421_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42421_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42421_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42421_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42421_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_48121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_48121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82221_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82221_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82221_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82221_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82221_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_84121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_84121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_161121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_162121_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162121_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162121_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162121_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_162121_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11422_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11422_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11422_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11422_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11422_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11822_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11822_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11822_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11822_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11822_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_111622_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111622_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111622_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111622_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111622_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12422_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12422_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12422_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12422_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12422_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12822_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12822_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12822_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12822_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12822_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14422_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14422_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14422_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14422_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14422_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_116122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21422_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21422_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21422_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21422_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21422_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21822_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21822_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21822_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21822_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21822_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22422_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22422_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22422_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22422_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22422_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41422_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41422_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41422_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41422_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41422_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81222_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81222_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81222_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81222_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81222_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_161122_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161122_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161122_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161122_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_161122_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11224_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11224_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11224_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11224_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11224_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11424_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11424_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11424_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11424_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11424_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11824_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11824_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11824_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11824_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11824_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12224_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12224_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12224_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12224_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12224_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12424_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12424_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12424_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12424_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12424_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14224_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14224_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14224_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14224_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14224_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21224_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21224_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21224_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21224_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21224_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21424_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21424_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21424_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21424_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21424_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22224_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22224_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22224_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22224_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22224_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41224_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41224_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41224_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41224_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41224_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81124_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81124_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81124_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81124_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81124_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11128_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11128_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11128_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11128_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11128_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11228_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11228_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11228_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11228_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11228_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11428_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11428_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11428_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11428_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11428_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12128_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12128_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12128_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12128_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12128_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12228_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12228_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12228_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12228_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12228_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14128_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14128_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14128_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14128_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14128_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21128_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21128_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21128_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21128_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21128_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21228_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21228_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21228_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21228_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21228_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22128_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22128_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22128_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22128_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22128_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41128_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41128_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41128_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41128_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41128_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11441_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11441_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11441_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11441_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11441_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11841_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11841_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11841_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11841_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11841_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12441_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12441_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12441_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12441_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12441_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12841_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12841_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12841_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12841_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12841_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14441_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14441_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14441_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14441_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14441_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_116141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_116141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21441_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21441_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21441_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21441_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21441_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22441_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22441_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22441_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22441_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22441_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_28141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_28141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42241_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42241_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42241_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42241_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42241_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_44141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_44141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_82141_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82141_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82141_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82141_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_82141_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11242_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11242_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11242_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11242_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11242_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11442_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11442_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11442_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11442_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11442_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11842_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11842_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11842_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11842_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11842_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12242_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12242_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12242_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12242_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12242_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12442_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12442_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12442_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12442_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12442_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14242_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14242_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14242_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14242_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14242_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21242_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21242_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21242_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21242_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21242_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21442_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21442_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21442_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21442_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21442_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22242_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22242_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22242_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22242_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22242_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41242_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41242_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41242_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41242_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41242_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_81142_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81142_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81142_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81142_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_81142_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11144_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11144_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11144_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11144_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11144_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11244_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11244_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11244_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11244_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11244_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11444_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11444_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11444_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11444_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11444_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12144_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12144_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12144_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12144_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12144_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12244_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12244_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12244_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12244_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12244_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14144_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14144_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14144_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14144_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14144_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21144_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21144_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21144_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21144_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21144_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21244_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21244_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21244_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21244_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21244_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22144_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22144_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22144_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22144_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22144_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41144_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41144_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41144_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41144_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41144_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11281_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11281_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11281_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11281_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11281_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11481_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11481_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11481_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11481_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11481_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12281_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12281_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12281_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12281_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12281_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12481_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12481_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12481_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12481_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12481_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14281_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14281_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14281_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14281_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14281_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_18181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_18181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21281_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21281_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21281_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21281_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21281_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22281_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22281_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22281_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22281_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22281_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_24181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_24181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_42181_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42181_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42181_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42181_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_42181_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11182_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11182_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11182_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11182_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11182_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11282_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11282_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11282_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11282_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11282_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_11482_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11482_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11482_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11482_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11482_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12182_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12182_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12182_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12182_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12182_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_12282_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12282_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12282_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12282_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12282_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_14182_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14182_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14182_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14182_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_14182_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21182_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21182_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21182_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21182_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21182_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_21282_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21282_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21282_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21282_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21282_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_22182_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22182_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22182_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22182_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_22182_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_41182_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41182_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41182_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41182_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_41182_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_111161_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111161_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111161_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111161_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111161_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_112161_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112161_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112161_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112161_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112161_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_121161_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121161_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121161_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121161_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121161_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_122161_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_122161_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_122161_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_122161_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_122161_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_141161_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_141161_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_141161_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_141161_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_141161_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_211161_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211161_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211161_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211161_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211161_sk2048(
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

at::Tensor f8f8bf16_rowwise_16x16x32_221161_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221161_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221161_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221161_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_221161_sk2048(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


