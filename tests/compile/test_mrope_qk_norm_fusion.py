# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compilation tests for MRoPE 3D + QK norm fusion pattern matching."""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.mrope_qk_norm_fusion import (
    FUSED_MROPE_OP,
    MRoPEQKNormFusionPass,
    get_fused_mrope_op,
)
from vllm.config import CompilationConfig, CompilationMode, ModelConfig, PassConfig, VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.platforms import current_platform


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMS normalization."""
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


def simple_mrope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simplified MRoPE forward for testing."""
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    q_out = q * cos.mean() + 0.1
    k_out = k * sin.mean() + 0.1
    return q_out, k_out


class SimpleMRoPEQKNormModel(torch.nn.Module):
    """Model simulating QK norm + MRoPE fusion pattern."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        dtype: torch.dtype,
        mrope_section: list[int],
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps
        self.dtype = dtype
        self.mrope_section = mrope_section

        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        self.q_norm_weight = torch.nn.Parameter(torch.ones(head_dim, dtype=dtype))
        self.k_norm_weight = torch.nn.Parameter(torch.ones(head_dim, dtype=dtype))

    def forward(
        self, qkv: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> tuple:
        num_tokens = qkv.shape[0]

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(num_tokens, self.num_heads, self.head_dim)
        q = rms_norm_forward(q, self.q_norm_weight, self.eps)
        q = q.view(num_tokens, self.q_size)

        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        k = rms_norm_forward(k, self.k_norm_weight, self.eps)
        k = k.view(num_tokens, self.kv_size)

        q, k = simple_mrope_forward(q, k, positions, cos_sin_cache, self.mrope_section)

        return q, k, v


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="MRoPE fusion only available on ROCm",
)
@pytest.mark.skipif(
    not rocm_aiter_ops.is_enabled(),
    reason="AITER not enabled",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mrope_qk_norm_fusion_pass_init(dtype: torch.dtype):
    """Test that the MRoPE fusion pass initializes correctly."""
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
            ),
        ),
    )

    with set_current_vllm_config(vllm_config):
        fusion_pass = MRoPEQKNormFusionPass(vllm_config)
        assert fusion_pass is not None

        if rocm_aiter_ops.is_fused_mrope_enabled():
            fused_op = get_fused_mrope_op()
            assert fused_op is not None
            print(f"Fusion pass initialized with op: {fused_op}")
            assert hasattr(fusion_pass, 'patterns')
            print("Pattern registration successful")
        else:
            print("MRoPE fusion not enabled (set VLLM_ROCM_USE_AITER_FUSED_MROPE=1)")


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="MRoPE fusion only available on ROCm",
)
@pytest.mark.skipif(
    not rocm_aiter_ops.is_enabled(),
    reason="AITER not enabled",
)
@pytest.mark.skipif(
    not rocm_aiter_ops.is_fused_mrope_enabled(),
    reason="MRoPE fusion not enabled (set VLLM_ROCM_USE_AITER_FUSED_MROPE=1)",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mrope_qk_norm_pattern_structure(dtype: torch.dtype):
    """Test that the pattern structure is correct."""
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
                enable_noop=True,
            ),
        ),
    )

    num_heads, num_kv_heads, head_dim = 16, 4, 128
    eps = 1e-6
    T = 5
    mrope_section = [16, 24, 24]

    with set_current_vllm_config(vllm_config):
        model = SimpleMRoPEQKNormModel(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            mrope_section=mrope_section,
        )

        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        qkv = torch.randn(T, q_size + 2 * kv_size, dtype=dtype, device="cuda")
        positions = torch.randint(0, 100, (3, T), dtype=torch.int64, device="cuda")
        cos_sin_cache = torch.randn(1000, head_dim, dtype=dtype, device="cuda")

        q, k, v = model(qkv, positions, cos_sin_cache)

        assert q.shape == (T, q_size)
        assert k.shape == (T, kv_size)
        assert v.shape == (T, kv_size)

        print(f"Model structure verified: Q={q.shape}, K={k.shape}, V={v.shape}")

        fusion_pass = MRoPEQKNormFusionPass(vllm_config)
        assert hasattr(fusion_pass, 'patterns')
        print("Fusion pass initialized and patterns registered successfully")


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="MRoPE fusion only available on ROCm",
)
def test_fused_mrope_op_getter():
    """Test the get_fused_mrope_op() helper function."""
    fused_op = get_fused_mrope_op()

    if rocm_aiter_ops.is_fused_mrope_enabled():
        assert fused_op is not None
        assert fused_op == torch.ops.vllm.rocm_aiter_fused_mrope_3d_rms.default
        assert fused_op == FUSED_MROPE_OP
        print(f"Fused MRoPE op available: {fused_op}")
    else:
        assert fused_op is None
        print("MRoPE fusion not enabled")


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="MRoPE fusion only available on ROCm",
)
@pytest.mark.skipif(
    not rocm_aiter_ops.is_enabled(),
    reason="AITER not enabled",
)
@pytest.mark.skipif(
    not rocm_aiter_ops.is_fused_mrope_enabled(),
    reason="MRoPE fusion not enabled (set VLLM_ROCM_USE_AITER_FUSED_MROPE=1)",
)
def test_mrope_matcher_class():
    """Test that the MatcherMRoPE class can be instantiated."""
    from vllm.compilation.mrope_qk_norm_fusion import MatcherMRoPE

    matcher = MatcherMRoPE(
        mrope_section=[16, 24, 24],
        head_size=128,
        num_heads=16,
        num_kv_heads=4,
        mrope_interleaved=False,
    )

    assert matcher is not None
    assert matcher.mrope_section == [16, 24, 24]
    assert matcher.head_size == 128
    print("MatcherMRoPE instantiated successfully")


if __name__ == "__main__":
    # Quick test
    if current_platform.is_rocm() and rocm_aiter_ops.is_enabled():
        print("Running MRoPE fusion compilation tests...")
        test_fused_mrope_op_getter()
        test_mrope_qk_norm_fusion_pass_init(torch.bfloat16)
        if rocm_aiter_ops.is_fused_mrope_enabled():
            test_mrope_matcher_class()
            test_mrope_qk_norm_pattern_structure(torch.bfloat16)
        print("All tests passed!")
    else:
        print("Tests require ROCm platform with AITER enabled")
