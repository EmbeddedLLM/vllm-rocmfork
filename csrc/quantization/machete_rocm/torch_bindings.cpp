#include "core/registration.h"
#include "ops.cu"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, machete_rocm_ops) {
    machete_rocm_ops.def("preshuffle(Tensor tensor, int Nxdl) -> (Tensor)", &machete_rocm::preshuffle);
    machete_rocm_ops.impl("preshuffle", torch::kCUDA, &machete_rocm::preshuffle);
    machete_rocm_ops.def("preshuffle_cpu(Tensor tensor, int Nxdl) -> (Tensor)", &machete_rocm::preshuffle_cpu);
    machete_rocm_ops.impl("preshuffle_cpu", torch::kCPU, &machete_rocm::preshuffle_cpu);

    machete_rocm_ops.def("scaled_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, int op_id, int kbatch) -> (Tensor)", &scaled_mm);
    machete_rocm_ops.impl("scaled_mm", torch::kCUDA, &scaled_mm);

}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)