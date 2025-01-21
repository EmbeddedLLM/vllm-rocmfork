#include "core/registration.h"
#include "ops.cuh"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, machete_rocm_ops) {
    machete_rocm_ops.def("preshuffle(Tensor tensor, int Nxdl) -> (Tensor)", &machete_rocm::preshuffle);
    machete_rocm_ops.impl("preshuffle", torch::kCUDA, &machete_rocm::preshuffle);
    machete_rocm_ops.def("preshuffle_cpu(Tensor tensor, int Nxdl) -> (Tensor)", &machete_rocm::preshuffle_cpu);
    machete_rocm_ops.impl("preshuffle_cpu", torch::kCPU, &machete_rocm::preshuffle_cpu);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)