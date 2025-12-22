# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AITER fused MRoPE 3D + QK norm kernel."""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMS normalization implementation."""
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Reference rotary embedding implementation."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings."""
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


def reference_fused_mrope_qk_norm(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    mrope_section: list[int],
    is_interleaved: bool,
) -> torch.Tensor:
    """Reference implementation of fused MRoPE 3D + QK norm."""
    num_tokens = qkv.shape[0]
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim

    # Split QKV
    qkv_flat = qkv.view(num_tokens, q_size + k_size + v_size)
    q, k, v = qkv_flat.split([q_size, k_size, v_size], dim=-1)

    # Apply RMS norm to Q and K
    q_by_head = q.view(num_tokens, num_heads_q, head_dim)
    q_by_head = rms_norm_forward(q_by_head, q_weight, eps)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(num_tokens, num_heads_k, head_dim)
    k_by_head = rms_norm_forward(k_by_head, k_weight, eps)
    k = k_by_head.view(k.shape)

    # Get cos/sin for positions (3D positions: T/H/W)
    # position_ids has shape (3, num_tokens)
    positions = position_ids.view(3, num_tokens)
    cos_sin = cos_sin_cache[positions]  # (3, num_tokens, head_dim)
    cos, sin = cos_sin.chunk(2, dim=-1)  # Each (3, num_tokens, head_dim // 2)

    # Apply MRoPE section processing
    if is_interleaved:
        cos = apply_interleaved_rope(cos, mrope_section)
        sin = apply_interleaved_rope(sin, mrope_section)
    else:
        # Concatenate sections: [T, H, W] -> [T0, H1, W2]
        cos = torch.cat(
            [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
            dim=-1,
        )
        sin = torch.cat(
            [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
            dim=-1,
        )

    # Apply rotary embedding to Q and K
    q_shape = q.shape
    q = q.view(num_tokens, -1, head_dim)
    q = apply_rotary_emb_torch(q, cos, sin, is_neox)
    q = q.reshape(q_shape)

    k_shape = k.shape
    k = k.view(num_tokens, -1, head_dim)
    k = apply_rotary_emb_torch(k, cos, sin, is_neox)
    k = k.reshape(k_shape)

    # Concatenate back to QKV format
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER MRoPE fusion only available on ROCm",
)
@pytest.mark.skipif(
    not rocm_aiter_ops.is_enabled(),
    reason="AITER not enabled",
)
@pytest.mark.skipif(
    not rocm_aiter_ops.is_fused_mrope_enabled(),
    reason="AITER MRoPE fusion not enabled (set VLLM_ROCM_USE_AITER_FUSED_MROPE=1)",
)
@pytest.mark.parametrize("num_tokens", [4, 8, 16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_interleaved", [True, False])
def test_aiter_fused_mrope_qk_norm_matches_reference(
    num_tokens: int,
    eps: float,
    is_neox: bool,
    dtype: torch.dtype,
    is_interleaved: bool,
):
    """Test AITER fused MRoPE + QK norm against reference implementation."""
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Configuration matching Qwen2-VL style
    num_heads_q = 16
    num_heads_k = 4
    num_heads_v = 4
    head_dim = 128
    max_positions = 4096
    mrope_section = [16, 24, 24]  # Typical for vision models: T/H/W sections

    # Verify section sizes sum to rotary_dim // 2
    rotary_dim = head_dim  # Assuming full rotation
    assert sum(mrope_section) == rotary_dim // 2

    device = torch.device("cuda")

    # Create test data
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim
    total_size = q_size + k_size + v_size

    # Input QKV tensor
    qkv_ref = torch.randn(num_tokens, total_size, dtype=dtype, device=device)
    qkv_aiter = qkv_ref.clone()

    # Norm weights
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)

    # Cos/sin cache (max_positions, head_dim)
    cos_sin_cache = torch.randn(max_positions, head_dim, dtype=dtype, device=device)

    # 3D positions for MRoPE (T/H/W)
    position_ids = torch.randint(
        0,
        max_positions // 4,  # Use smaller range to avoid OOB
        (3, num_tokens),
        dtype=torch.int64,
        device=device,
    )

    # Run reference implementation
    qkv_ref_out = reference_fused_mrope_qk_norm(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        mrope_section,
        is_interleaved,
    )

    # Run AITER kernel (in-place modification)
    rocm_aiter_ops.fused_mrope_3d_rms(
        qkv_aiter,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        mrope_section,
        is_interleaved,
    )

    # Compare results
    # Use relaxed tolerances for fused kernels with bfloat16
    if dtype == torch.bfloat16:
        atol, rtol = 5e-2, 1e-2
    else:
        atol, rtol = 5e-2, 1e-2

    torch.testing.assert_close(
        qkv_aiter,
        qkv_ref_out,
        atol=atol,
        rtol=rtol,
        msg=f"AITER MRoPE fusion output differs from reference "
        f"(num_tokens={num_tokens}, eps={eps}, is_neox={is_neox}, "
        f"dtype={dtype}, is_interleaved={is_interleaved})",
    )

    print(
        f"✅ Test passed: num_tokens={num_tokens}, eps={eps}, is_neox={is_neox}, "
        f"dtype={dtype}, is_interleaved={is_interleaved}"
    )
