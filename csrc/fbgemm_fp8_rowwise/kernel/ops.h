
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


at::Tensor f8f8bf16_rowwise_32x32x16_111132(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111132(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_211132(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211132(
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


at::Tensor f8f8bf16_rowwise_32x32x16_111216(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111216(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_112216(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112216(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_121216(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121216(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_211216(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211216(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111232(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111232(
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


at::Tensor f8f8bf16_rowwise_32x32x16_11148(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11148(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11248(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11248(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12148(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12148(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21148(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21148(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111416(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111416(
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


at::Tensor f8f8bf16_rowwise_32x32x16_11184(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11184(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11284(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11284(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_12184(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_12184(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_21184(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_21184(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_11188(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_11188(
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


at::Tensor f8f8bf16_rowwise_32x32x16_111162(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111162(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_112162(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_112162(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_121162(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121162(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_211162(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_211162(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111164(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111164(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111321(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111321(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_121321(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_121321(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


at::Tensor f8f8bf16_rowwise_32x32x16_111322(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor f8f8bf16_rowwise_16x16x32_111322(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);


