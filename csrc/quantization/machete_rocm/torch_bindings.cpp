#include "core/registration.h"
#include "quantization/machete_rocm/ops.h"

#include "generated/machete_mm_kernel.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, machete_rocm_ops) {
    // Machete fp8 Rowwise Scaling GEMM ops
    machete_rocm_ops.def("prepackB_cpu_rocm(Tensor B, const int NXdl=32) -> (Tensor)", &prepackB_cpu);
    machete_rocm_ops.impl("prepackB_cpu_rocm", torch::kCUDA, &prepackB_cpu);

}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)