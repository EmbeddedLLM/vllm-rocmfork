# Machete ROCm (FP8 CK-Based GEMM)

Machete ROCm is design in the spirit of Machete kernel but optimized for MI300X architectures and based on Composable Kernel (CK).

## Overview

Machete ROCm currently only supports Rowwise Scaled GEMM, which effectively performs

```
A = torch.rand((M, K), dtype=torch.float16).cuda().to(torch.float8_e4m3fnuz)
B = torch.rand((N, K), dtype=torch.float16).cuda().to(dtype)
a_scale = torch.rand((M, 1), dtype=torch.float32).cuda()
b_scale = torch.rand((N, 1), dtype=torch.float32).cuda()
scale_a_dummy = torch.tensor(1.0, device="cuda", dtype=torch.float32)
scale_b_dummy = torch.tensor(1.0, device="cuda", dtype=torch.float32)
scaled_mm_output = torch._scaled_mm(
    A.to(torch.float8_e4m3fnuz), B.cuda().t(), scale_a_dummy, scale_b_dummy, out_dtype=torch.float16
)
scaled_mm_output = scaled_mm_output * a_scale * b_scale.t()
```

Supported DataType

| A                 | B                 | a_scale   | b_scale | output  |
|---                |---                | --------  | ------- | ------  |
| float8_e4m3fnuz   | float8_e4m3fnuz   | float32   | float32 | float16 |

## API

The main optimization within Machete is prepacking the weight matrix to more closely match the tensor core layouts, allowing for wider shared memory loads when loading the weight matrix. This means that the weight matrix must be prepacked before calling `machete_gemm`. The flow looks something like:

```
from vllm import _custom_ops as ops

...
W_q_packed = ops.machete_prepack_B_rocm(w_q, wtype)
output = ops.machete_gemm_rocm(
    a,
    b_q=W_q_packed,
    b_type=wtype,
    b_scales=w_s,
    b_group_size=group_size
)
```

## Code Generation

Since Machete is based on Composable Kernel, we can generate multiple type pairs and different tile shapes using the same kernel template. We generate multiple instantiations of this template using `generate.py`. 
