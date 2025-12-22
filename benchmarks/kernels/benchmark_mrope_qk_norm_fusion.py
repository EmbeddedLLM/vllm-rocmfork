#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark for AITER fused MRoPE 3D + QK norm kernel.

Compares performance of:
1. Unfused PyTorch implementation (reference)
2. AITER fused MRoPE 3D + QK norm kernel

Usage:
    # With default parameters
    python benchmarks/kernels/benchmark_mrope_qk_norm_fusion.py

    # With custom parameters
    python benchmarks/kernels/benchmark_mrope_qk_norm_fusion.py \\
        --num-tokens 256 --num-heads 32 --num-kv-heads 8 --head-dim 128

    # Enable AITER kernel
    VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSED_MROPE=1 \\
    python benchmarks/kernels/benchmark_mrope_qk_norm_fusion.py
"""

import argparse
import os
from typing import Callable

import torch
import triton

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


def apply_mrope_unfused(
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
    """Unfused PyTorch implementation of MRoPE 3D + QK norm."""
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

    # Get cos/sin for 3D positions
    positions = position_ids.view(3, num_tokens)
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    # Apply MRoPE section processing
    if is_interleaved:
        cos = apply_interleaved_rope(cos, mrope_section)
        sin = apply_interleaved_rope(sin, mrope_section)
    else:
        cos = torch.cat(
            [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
            dim=-1,
        )
        sin = torch.cat(
            [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
            dim=-1,
        )

    # Apply rotary embedding
    q_shape = q.shape
    q = q.view(num_tokens, -1, head_dim)
    q = apply_rotary_emb_torch(q, cos, sin, is_neox)
    q = q.reshape(q_shape)

    k_shape = k.shape
    k = k.view(num_tokens, -1, head_dim)
    k = apply_rotary_emb_torch(k, cos, sin, is_neox)
    k = k.reshape(k_shape)

    # Concatenate back
    return torch.cat([q, k, v], dim=-1)


def apply_mrope_aiter(
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
) -> None:
    """AITER fused MRoPE 3D + QK norm (in-place)."""
    rocm_aiter_ops.fused_mrope_3d_rms(
        qkv,
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


def calculate_diff(
    a: torch.Tensor, b: torch.Tensor, atol: float = 5e-2, rtol: float = 1e-2
) -> tuple[float, bool]:
    """Calculate difference and check if within tolerance."""
    diff = torch.abs(a - b)
    max_diff = diff.max().item()
    # Use torch.testing.assert_close logic for consistency with unit tests
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        matches = True
    except AssertionError:
        matches = False
    return max_diff, matches


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[64, 256, 1024, 4096],
        line_arg="provider",
        line_vals=["unfused", "aiter"],
        line_names=["Unfused (PyTorch)", "AITER Fused"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="Time (μs)",
        plot_name="MRoPE 3D + QK Norm Fusion Benchmark",
        args={},
    )
)
def benchmark_mrope_fusion(
    num_tokens: int,
    provider: str,
    num_heads_q: int = 16,
    num_heads_k: int = 4,
    num_heads_v: int = 4,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-6,
    is_neox: bool = True,
    is_interleaved: bool = False,
):
    """Benchmark MRoPE fusion implementations."""
    device = torch.device("cuda")
    max_positions = 4096
    mrope_section = [16, 24, 24]  # T/H/W sections

    # Create test data
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim
    total_size = q_size + k_size + v_size

    qkv = torch.randn(num_tokens, total_size, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    cos_sin_cache = torch.randn(max_positions, head_dim, dtype=dtype, device=device)
    position_ids = torch.randint(
        0, max_positions // 4, (3, num_tokens), dtype=torch.int64, device=device
    )

    # Select implementation
    if provider == "unfused":
        fn = lambda: apply_mrope_unfused(
            qkv.clone(),
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
    elif provider == "aiter":
        if not rocm_aiter_ops.is_fused_mrope_enabled():
            return float("inf")  # Skip if not available
        fn = lambda: apply_mrope_aiter(
            qkv.clone(),
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
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return ms * 1000  # Convert to microseconds


def main():
    parser = argparse.ArgumentParser(description="Benchmark MRoPE 3D + QK norm fusion")
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=256,
        help="Number of tokens (default: 256)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Number of query heads (default: 16)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=4,
        help="Number of key/value heads (default: 4)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dimension (default: 128)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./benchmark_results",
        help="Path to save benchmark results (default: ./benchmark_results)",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=" * 80)
    print("MRoPE 3D + QK Norm Fusion Benchmark")
    print("=" * 80)
    print(f"Platform: {current_platform.get_device_name()}")
    print(f"Number of tokens: {args.num_tokens}")
    print(f"Number of heads (Q): {args.num_heads}")
    print(f"Number of heads (K/V): {args.num_kv_heads}")
    print(f"Head dimension: {args.head_dim}")
    print(f"Data type: {args.dtype}")
    print(f"AITER enabled: {rocm_aiter_ops.is_enabled()}")
    print(f"AITER MRoPE fusion enabled: {rocm_aiter_ops.is_fused_mrope_enabled()}")
    print()

    # Correctness test
    print("=" * 80)
    print("Correctness Test")
    print("=" * 80)

    device = torch.device("cuda")
    max_positions = 4096
    mrope_section = [16, 24, 24]
    eps = 1e-6
    is_neox = True
    is_interleaved = False

    q_size = args.num_heads * args.head_dim
    k_size = args.num_kv_heads * args.head_dim
    v_size = args.num_kv_heads * args.head_dim
    total_size = q_size + k_size + v_size

    # Create test data
    torch.manual_seed(42)
    qkv_ref = torch.randn(args.num_tokens, total_size, dtype=dtype, device=device)
    q_weight = torch.randn(args.head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(args.head_dim, dtype=dtype, device=device)
    cos_sin_cache = torch.randn(max_positions, args.head_dim, dtype=dtype, device=device)
    position_ids = torch.randint(
        0, max_positions // 4, (3, args.num_tokens), dtype=torch.int64, device=device
    )

    # Run unfused reference
    qkv_unfused = apply_mrope_unfused(
        qkv_ref.clone(),
        args.num_heads,
        args.num_kv_heads,
        args.num_kv_heads,
        args.head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        mrope_section,
        is_interleaved,
    )

    # Run AITER if available
    if rocm_aiter_ops.is_fused_mrope_enabled():
        qkv_aiter = qkv_ref.clone()
        apply_mrope_aiter(
            qkv_aiter,
            args.num_heads,
            args.num_kv_heads,
            args.num_kv_heads,
            args.head_dim,
            eps,
            q_weight,
            k_weight,
            cos_sin_cache,
            is_neox,
            position_ids,
            mrope_section,
            is_interleaved,
        )

        # Check correctness
        max_diff, matches = calculate_diff(qkv_aiter, qkv_unfused)
        status = "✅ Matches" if matches else "❌ Differs"
        print(f"AITER kernel: {status} (max diff: {max_diff:.6f})")
    else:
        print("AITER kernel: ⚠️  Not available (enable with VLLM_ROCM_USE_AITER_FUSED_MROPE=1)")

    print()

    # Performance benchmark
    print("=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    # Run benchmark
    benchmark_mrope_fusion.run(
        print_data=True,
        save_path=args.save_path,
        num_heads_q=args.num_heads,
        num_heads_k=args.num_kv_heads,
        num_heads_v=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
