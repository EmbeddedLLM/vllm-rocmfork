#pragma once

#include <torch/all.h>

at::Tensor prepackB(at::Tensor B, const int NXdl=32);
at::Tensor prepackB_cpu(at::Tensor B, const int NXdl=32);

// definitions of benchmarking ops
#include "generated/machete_mm_kernel.h"
    