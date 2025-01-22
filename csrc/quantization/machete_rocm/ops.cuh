#pragma once
#include <torch/torch.h>


at::Tensor scaled_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t op_id,
    int64_t kbatch
);
namespace machete_rocm{

    at::Tensor preshuffle(
        const at::Tensor tensor,
        const int64_t Nxdl
    );

    at::Tensor preshuffle_cpu(
        const at::Tensor tensor,
        const int64_t Nxdl
    );

}