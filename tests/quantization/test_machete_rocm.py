from typing import Tuple
import pytest
import torch

from tests.quantization.utils import (
    is_quant_method_supported,
)
import vllm._machete_rocm_C  # noqa: F401

PREPACK_TENSOR_TEST_SIZE_LIST = [
    (64, 64),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
    (4096, 2048),
    (64, 63),
    (8192, 7392),
    (8192, 8128),

]

@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("size", PREPACK_TENSOR_TEST_SIZE_LIST)
def test_ck_preshuffle_cpu_equal_gpu(size: Tuple[int, int]) -> None:
    NXdl= 32
    cpu_tensor = torch.rand(size, dtype=torch.float16).to(torch.float8_e4m3fnuz)
    cpu_tensor_shuffled = torch.ops._machete_rocm_C.preshuffle_cpu(cpu_tensor, NXdl)

    device_tensor = cpu_tensor.cuda()
    device_tensor_shuffled= torch.ops._machete_rocm_C.preshuffle(device_tensor, NXdl)

    assert torch.allclose(
        cpu_tensor_shuffled.to(torch.float16),
        device_tensor_shuffled.to(torch.float16).cpu()
    )



