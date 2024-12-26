import torch
import pytest
from typing import Type, Optional
import numpy as np

import vllm._fp8gemm_C # noqa: F401
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

from tests.kernels.utils import opcheck


device = "cuda"
if not current_platform.is_rocm():
    pytest.skip(reason="FBGEMM Kernel currently only supported on ROCm through CK kernel.",
                allow_module_level=True)
    
def get_cond_label(M, N, K):
    if M < 64 and N < 2048 and K < 2048:  # COND_1
        return "COND_1"
    elif M < 64 and K < 2048:  # COND_2
        return "COND_2"
    elif M < 64 and N < 2048:  # COND_3
        return "COND_3"
    # elif M < 64 and N > 2048 and K > 2048: # COND_4
    #     return "COND_4"
    elif M < 64:  # COND_5
        return "COND_5"
    elif (
        (M < 512 and K < 8192) or (N <= 2048 and K <= 8192) or (K <= 2048 and N <= 8192)
    ) and K >= 1024:  # COND_6
        return "COND_6"
    elif K < 1024:  # COND_7
        return "COND_7"
    elif M < 1024:  # COND_8
        return "COND_8"
    elif M >= 1024 and N >= 1024 and K >= 1024:  # COND_9
        return "COND_9"
    else:  # COND_10 Will never be triggered
        return "COND_10"

def vec_scaled_mm_torch(a: torch.Tensor,
                    b: torch.Tensor,
                    scale_a: torch.Tensor,
                    scale_b: torch.Tensor,
                    out_dtype: Type[torch.dtype],
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out

def generate_random_fp8_tensor(shape):
    # Generate random float32 tensor and convert to float8_e4m3fnuz
    tensor = torch.rand(shape, dtype=torch.float32, device=device)
    return tensor.to(torch.float8_e4m3fnuz)

def generate_random_scale_tensor(shape):
    return torch.rand(shape, dtype=torch.float32, device=device)

def _get_opcheck_params():
    
    MNK_list = []
    for M in range(17):
        for N in range(17):
            for K in range(17):
                # cond_str = get_cond_label(2**M, 2**N, 2**K)
                # if cond_str in ["COND_10", "COND_2"]:
                #     print(cond_str, "M, N, K ", 2**M, 2**N, 2**K)
                MNK_list.append((2**M, 2**N, 2**K))

    return MNK_list

@pytest.mark.parametrize("M, N, K",
    [
    (56, 8192, 7392), # Qwen Fail Case
    (16, 1280, 1024),  # Case 0 COND_1
    (16, 1280, 8192),  # Case 1 COND_3
    (32, 1280, 8192),  # Case 2 COND_3
    (64, 1280, 8192),  # Case 3 COND_6
    (128, 1280, 8192),  # Case 4 COND_6
    (16, 8192, 1024),  # Case 5 COND_2
    (32, 8192, 1024),  # Case 6 COND_2
    (64, 8192, 1024),  # Case 7 COND_6
    (128, 8192, 1024),  # Case 8 COND_6
    (16, 7168, 8192),  # Case 9 COND_5
    (32, 7168, 8192),  # Case 10 COND_5
    (64, 7168, 8192),  # Case 11 COND_8
    (128, 7168, 8192),  # Case 12 COND_8
    (1024, 7168, 8192),  # Case 13 COND_9
    (2048, 7168, 8192),  # Case 14 COND_9
    (4096, 7168, 8192),  # Case 15 COND_9
    (8192, 7168, 8192),  # Case 16 COND_9
    (16, 8192, 3584),  # Case 17 COND_5
    (32, 8192, 3584),  # Case 18 COND_5
    (64, 8192, 3584),  # Case 19 COND_6
    (128, 8192, 3584),  # Case 20 COND_6
    (1024, 8192, 3584),  # Case 21 COND_9
    (2048, 8192, 3584),  # Case 22 COND_9
    (4096, 8192, 3584),  # Case 23 COND_9
    (8192, 8192, 3584),  # Case 24 COND_9
    (32, 13312, 6656),  # Case 25 COND_5
    (64, 13312, 6656),  # Case 26 COND_6
    (128, 13312, 6656),  # Case 27 COND_6
    (16, 13312, 16384),  # Case 28 COND_5
    (32, 13312, 16384),  # Case 29 COND_5
    (64, 13312, 16384),  # Case 30 COND_8
    (128, 13312, 16384),  # Case 31 COND_8
    (1024, 13312, 16384),  # Case 32 COND_9
    (2048, 13312, 16384),  # Case 33 COND_9
    (4096, 13312, 16384),  # Case 34 COND_9
    (8192, 13312, 16384),  # Case 35 COND_9
    (32, 16384, 6656),  # Case 36 COND_5
    (64, 16384, 6656),  # Case 37 COND_6
    (128, 16384, 6656),  # Case 38 COND_6
    (1024, 16384, 6656),  # Case 39 COND_9
    (2048, 16384, 6656),  # Case 40 COND_9
    (4096, 16384, 6656),  # Case 41 COND_9
    (8192, 16384, 6656),  # Case 42 COND_9
    (16, 16384, 16384),  # Case 43 COND_5
    (32, 16384, 16384),  # Case 44 COND_5
    (64, 16384, 16384),  # Case 45 COND_8
    (128, 16384, 16384),  # Case 46 COND_8
    (1536, 3584, 3584),  # Case 47 COND_9
    (8192, 9728, 3584),  # Case 48 COND_9
    (8192, 3584, 9728),  # Case 49 COND_9
    (8192, 3584, 3584),  # Case 50 COND_9
    (4096, 3584, 3584),  # Case 51 COND_9
    (768, 3584, 3584),  # Case 52 COND_8
    (4096, 9728, 3584),  # Case 53 COND_9
    (4096, 3584, 9728),  # Case 54 COND_9
    (7200, 3584, 3584),  # Case 55 COND_9
    (7200, 9728, 3584),  # Case 56 COND_9
    (7200, 3584, 9728),  # Case 57 COND_9
    (3600, 3584, 3584),  # Case 58 COND_9
    (3600, 9728, 3584),  # Case 59 COND_9
    (3600, 3584, 9728),  # Case 60 COND_9
    (1536, 4096, 4096),  # Case 61 COND_9
    (3600, 4096, 4096),  # Case 62 COND_9
    (3600, 11008, 4096),  # Case 63 COND_9
    (3600, 4096, 11008),  # Case 64 COND_9
    (4096, 4096, 4096),  # Case 65 COND_9
    (4096, 11008, 4096),  # Case 66 COND_9
    (4096, 4096, 11008),  # Case 67 COND_9
    (32768, 128, 8192),  # Case 68 COND_6
    (32768, 8192, 1024),  # Case 69 COND_6
    (32768, 8192, 3072),  # Case 70 COND_9
    (32768, 3072, 8192),  # Case 71 COND_9
    (32768, 1024, 8192),  # Case 72 COND_6
    (512, 2048, 1000), # COND_7 FAILED
    (1024, 512, 512), # COND_7
    (512, 204, 512), # COND_7 FAILED
    (512, 512, 2048), # COND_6
    (4, 2048, 1024), # COND_2
    (2, 16384, 1024), # COND_2
    (1, 32768, 1), # COND_2 FAILED
    (32, 16384, 1024), # COND_2
    (1024, 1, 16384), # COND_10 FAILED
    (1024, 512, 32768), # COND_10
    (32768, 512, 32768), # COND_10             
    ] )
def test_f8f8bf16_rowwise_opcheck(M, N, K):
    # Generate random input tensors
    XQ = generate_random_fp8_tensor((M, K))
    WQ = generate_random_fp8_tensor((N, K))
    x_scale = generate_random_scale_tensor((M, 1))
    w_scale = generate_random_scale_tensor((N, 1))
    # print(M, N, K, get_cond_label(M, N, K))

    # f8f8bf16_rowwise expect
    # X = (M, K)
    # W = (N, K)
    output = opcheck(
        torch.ops._fp8gemm_C.f8f8bf16_rowwise,
        (XQ, WQ, x_scale, w_scale, None, True),
        test_utils="test_schema"
    )


@pytest.mark.parametrize("M, N, K", [
    (56, 8192, 7392), # Qwen Fail Case
    (16, 1280, 1024),  # Case 0 COND_1
    (16, 1280, 8192),  # Case 1 COND_3
    (32, 1280, 8192),  # Case 2 COND_3
    (64, 1280, 8192),  # Case 3 COND_6
    (128, 1280, 8192),  # Case 4 COND_6
    (16, 8192, 1024),  # Case 5 COND_2
    (32, 8192, 1024),  # Case 6 COND_2
    (64, 8192, 1024),  # Case 7 COND_6
    (128, 8192, 1024),  # Case 8 COND_6
    (16, 7168, 8192),  # Case 9 COND_5
    (32, 7168, 8192),  # Case 10 COND_5
    (64, 7168, 8192),  # Case 11 COND_8
    (128, 7168, 8192),  # Case 12 COND_8
    (1024, 7168, 8192),  # Case 13 COND_9
    (2048, 7168, 8192),  # Case 14 COND_9
    (4096, 7168, 8192),  # Case 15 COND_9
    (8192, 7168, 8192),  # Case 16 COND_9
    (16, 8192, 3584),  # Case 17 COND_5
    (32, 8192, 3584),  # Case 18 COND_5
    (64, 8192, 3584),  # Case 19 COND_6
    (128, 8192, 3584),  # Case 20 COND_6
    (1024, 8192, 3584),  # Case 21 COND_9
    (2048, 8192, 3584),  # Case 22 COND_9
    (4096, 8192, 3584),  # Case 23 COND_9
    (8192, 8192, 3584),  # Case 24 COND_9
    (32, 13312, 6656),  # Case 25 COND_5
    (64, 13312, 6656),  # Case 26 COND_6
    (128, 13312, 6656),  # Case 27 COND_6
    (16, 13312, 16384),  # Case 28 COND_5
    (32, 13312, 16384),  # Case 29 COND_5
    (64, 13312, 16384),  # Case 30 COND_8
    (128, 13312, 16384),  # Case 31 COND_8
    (1024, 13312, 16384),  # Case 32 COND_9
    (2048, 13312, 16384),  # Case 33 COND_9
    (4096, 13312, 16384),  # Case 34 COND_9
    (8192, 13312, 16384),  # Case 35 COND_9
    (32, 16384, 6656),  # Case 36 COND_5
    (64, 16384, 6656),  # Case 37 COND_6
    (128, 16384, 6656),  # Case 38 COND_6
    (1024, 16384, 6656),  # Case 39 COND_9
    (2048, 16384, 6656),  # Case 40 COND_9
    (4096, 16384, 6656),  # Case 41 COND_9
    (8192, 16384, 6656),  # Case 42 COND_9
    (16, 16384, 16384),  # Case 43 COND_5
    (32, 16384, 16384),  # Case 44 COND_5
    (64, 16384, 16384),  # Case 45 COND_8
    (128, 16384, 16384),  # Case 46 COND_8
    (1536, 3584, 3584),  # Case 47 COND_9
    (8192, 9728, 3584),  # Case 48 COND_9
    (8192, 3584, 9728),  # Case 49 COND_9
    (8192, 3584, 3584),  # Case 50 COND_9
    (4096, 3584, 3584),  # Case 51 COND_9
    (768, 3584, 3584),  # Case 52 COND_8
    (4096, 9728, 3584),  # Case 53 COND_9
    (4096, 3584, 9728),  # Case 54 COND_9
    (7200, 3584, 3584),  # Case 55 COND_9
    (7200, 9728, 3584),  # Case 56 COND_9
    (7200, 3584, 9728),  # Case 57 COND_9
    (3600, 3584, 3584),  # Case 58 COND_9
    (3600, 9728, 3584),  # Case 59 COND_9
    (3600, 3584, 9728),  # Case 60 COND_9
    (1536, 4096, 4096),  # Case 61 COND_9
    (3600, 4096, 4096),  # Case 62 COND_9
    (3600, 11008, 4096),  # Case 63 COND_9
    (3600, 4096, 11008),  # Case 64 COND_9
    (4096, 4096, 4096),  # Case 65 COND_9
    (4096, 11008, 4096),  # Case 66 COND_9
    (4096, 4096, 11008),  # Case 67 COND_9
    (32768, 128, 8192),  # Case 68 COND_6
    (32768, 8192, 1024),  # Case 69 COND_6
    (32768, 8192, 3072),  # Case 70 COND_9
    (32768, 3072, 8192),  # Case 71 COND_9
    (32768, 1024, 8192),  # Case 72 COND_6
    (512, 2048, 1000), # COND_7 FAILED
    (1024, 512, 512), # COND_7
    (512, 204, 512), # COND_7 FAILED
    (512, 512, 2048), # COND_6
    (4, 2048, 1024), # COND_2
    (2, 16384, 1024), # COND_2
    (1, 32768, 1), # COND_2 FAILED
    (32, 16384, 1024), # COND_2
    (1024, 1, 16384), # COND_10 FAILED
    (1024, 512, 32768), # COND_10
    (32768, 512, 32768), # COND_10
])
def test_f8f8bf16_rowwise(M, N, K):

    print(M, N, K, get_cond_label(M, N, K))
    
    # Generate random input tensors
    XQ = generate_random_fp8_tensor((M, K))
    WQ = generate_random_fp8_tensor((N, K))
    x_scale = generate_random_scale_tensor((M, 1))
    w_scale = generate_random_scale_tensor((N, 1))

    # Call the rowwise function
    # vec_scaled_mm_torch expect
    # X = (M, K)
    # W = (K, N)
    ref_output = vec_scaled_mm_torch(XQ, WQ.transpose(1,0), x_scale, w_scale, out_dtype=torch.bfloat16)

    # f8f8bf16_rowwise expect
    # X = (M, K)
    # W = (N, K)
    output = torch.ops._fp8gemm_C.f8f8bf16_rowwise(XQ, WQ, x_scale, w_scale, None, True)

    # Verify the output shape and dtype
    assert output.shape == (M, N)
    assert ref_output.shape == (M, N)
    assert output.dtype == torch.bfloat16
    assert ref_output.dtype == torch.bfloat16
    rtol, atol = (3e-2, 1e-3)
    
    assert torch.allclose(output, ref_output, rtol=rtol, atol=atol)

# def test_f8f8bf16_rowwise_out(M, N, K):
#     # Generate random input tensors
#     XQ = generate_random_fp8_tensor((M, K))
#     WQ = generate_random_fp8_tensor((N, K))
#     x_scale = generate_random_scale_tensor((M, 1))
#     w_scale = generate_random_scale_tensor((N, 1))
#     output = torch.empty((M, N), dtype=torch.bfloat16, device=device)

#     # Call the rowwise_out function
#     f8f8bf16_rowwise_out(XQ, WQ, x_scale, w_scale, output, None, True)

#     # Verify the output shape and dtype
#     assert output.shape == (M, N)
#     assert output.dtype == torch.bfloat16

if __name__ == "__main__":
    # pytest.main([__file__])
    print("RUN test")

    for M in range(16):
        for N in range(16):
            for K in range(16):
                cond_str = get_cond_label(2**M, 2**N, 2**K)
                if cond_str in ["COND_10", "COND_2"]:
                    print(cond_str, "M, N, K ", 2**M, 2**N, 2**K)