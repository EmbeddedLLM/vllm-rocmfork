#pragma once
#include <torch/torch.h>

namespace machete_rocm{
    namespace utils{
        at::Tensor align_to_wavefront(const at::Tensor tensor);
    }
}