#pragma once
#include <torch/torch.h>


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