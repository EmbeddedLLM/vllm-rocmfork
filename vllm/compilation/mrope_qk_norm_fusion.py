# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fusion pass for MRoPE 3D + QK norm operations.

This pass fuses the following pattern:
1. Split QKV tensor into Q, K, V
2. Apply RMS normalization to Q and K
3. Apply 3D Multi-RoPE (MRoPE) to Q and K with T/H/W sections
4. Concatenate Q, K, V back together

The fused kernel combines these operations for better performance.
"""

from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.platforms import current_platform

from .fusion import empty_bf16, empty_fp32, empty_i64
from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherRMSNorm
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


def get_fused_mrope_op():
    """Get the appropriate fused MRoPE kernel based on environment.

    Returns the AITER fused MRoPE kernel if available and enabled,
    otherwise returns None (no CUDA equivalent exists yet).
    """
    if (
        current_platform.is_rocm()
        and rocm_aiter_ops.is_fused_mrope_enabled()
        and hasattr(torch.ops.vllm, "rocm_aiter_fused_mrope_3d_rms")
    ):
        return torch.ops.vllm.rocm_aiter_fused_mrope_3d_rms.default
    return None


# Module-level constant for backward compatibility
FUSED_MROPE_OP = get_fused_mrope_op()


class MatcherMRoPE:
    """Matcher for MRoPE (Multi-Rope) operations.

    This matcher handles the 3D positional encoding pattern used in
    vision-language models like Qwen2-VL.
    """

    def __init__(
        self,
        mrope_section: list[int],
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        mrope_interleaved: bool = False,
    ):
        self.mrope_section = mrope_section
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.mrope_interleaved = mrope_interleaved

    def __call__(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match MRoPE pattern for tracing."""
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

        q_rope, k_rope = triton_mrope(
            q,
            k,
            cos,
            sin,
            self.mrope_section,
            self.head_size,
            self.head_size,
            self.mrope_interleaved,
        )
        return q_rope, k_rope


class MRoPEQKNormPattern:
    """Pattern matcher for MRoPE + QK Norm fusion.

    Matches the unfused sequence in attention blocks and replaces with the fused op.

    Unfused (conceptually):
      q, k, v = split(qkv, [qsz, kvsz, kvsz], -1)
      qh = reshape(q, [-1, num_heads, head_dim])
      kh = reshape(k, [-1, num_kv_heads, head_dim])
      qn = rms_norm(qh, q_weight, eps)
      kn = rms_norm(kh, k_weight, eps)
      qf = reshape(qn, [-1, num_heads * head_dim])
      kf = reshape(kn, [-1, num_kv_heads * head_dim])
      cos_sin = cos_sin_cache[positions]  # positions.shape = (3, T)
      cos, sin = cos_sin.chunk(2, dim=-1)
      qf, kf = triton_mrope(qf, kf, cos, sin, mrope_section, ...)
      return qf, kf, v

    Fused replacement:
      fused_mrope_3d_rms(qkv, num_heads, num_kv_heads, num_kv_heads, head_dim,
                         eps, q_weight, k_weight, cos_sin_cache, is_neox,
                         positions, mrope_section, is_interleaved)
      return split(qkv, [qsz, kvsz, kvsz], -1)
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.is_neox = is_neox
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        self.rmsnorm_matcher = MatcherRMSNorm(eps)
        self.mrope_matcher = MatcherMRoPE(
            mrope_section=mrope_section,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            mrope_interleaved=mrope_interleaved,
        )

        self.fused_op = get_fused_mrope_op()

    def get_inputs(self):
        """Sample inputs for pattern tracing."""
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        positions = empty_i64(3, T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
        cos_sin_cache = empty_bf16(4096, self.head_dim)
        return [
            qkv,
            positions,
            q_weight,
            k_weight,
            cos_sin_cache,
        ]

    @staticmethod
    def wrap_trace_fn(trace_fn, *process_fx_fns: Callable[[fx.GraphModule], None]):
        """Wrap trace function with additional processing."""

        def wrapped(*args, **kwargs):
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)
            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule):
        """Convert view operations to reshape for pattern matching."""
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    def register(self, pm_pass: PatternMatcherPass):
        """Register the pattern and replacement with the pattern matcher."""

        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_normed_by_head = self.rmsnorm_matcher(q_by_head, q_weight)
            q_flat = q_normed_by_head.view(q.shape)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_normed_by_head = self.rmsnorm_matcher(k_by_head, k_weight)
            k_flat = k_normed_by_head.view(k.shape)

            q_rope, k_rope = self.mrope_matcher(positions, q_flat, k_flat, cos_sin_cache)
            return q_rope, k_rope, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ):
            result = auto_functionalized(
                self.fused_op,
                qkv=qkv,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.eps,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                position_ids=positions,
                mrope_section=self.mrope_section,
                is_interleaved=self.mrope_interleaved,
            )
            result_qkv = result[1]
            return result_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            MRoPEQKNormPattern.wrap_trace_fn(
                pm.fwd_only,
                MRoPEQKNormPattern.fx_view_to_reshape,
            ),
            pm_pass,
        )


class MRoPEQKNormFusionPass(VllmPatternMatcherPass):
    """Fusion pass for MRoPE 3D + QK norm operations.

    Fuses Q/K RMSNorm + MRoPE into fused_mrope_3d_rms when the custom op exists.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="mrope_qk_norm_fusion_pass"
        )

        if not current_platform.is_rocm():
            logger.debug("MRoPE fusion only available on ROCm platform")
            return

        if not rocm_aiter_ops.is_fused_mrope_enabled():
            logger.debug(
                "MRoPE fusion not enabled. "
                "Set VLLM_ROCM_USE_AITER_FUSED_MROPE=1 to enable."
            )
            return

        fused_op = get_fused_mrope_op()
        if fused_op is None:
            logger.debug("Fused MRoPE kernel not available")
            return

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "MRoPE+QK Norm fusion not enabled: unsupported dtype %s", dtype
            )
            return

        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(config, Attention)
        if len(attn_layers) == 0:
            logger.warning_once(
                "MRoPE+QK Norm fusion enabled, but no Attention layers were discovered."
            )
            return

        layer = next(iter(attn_layers.values()))

        if not MRotaryEmbedding.enabled():
            logger.debug("MRotaryEmbedding not enabled, skipping MRoPE fusion")
            return

        for epsilon in [1e-5, 1e-6]:
            for neox in [True, False]:
                for interleaved in [False, True]:
                    for mrope_section in [[16, 24, 24], [8, 16, 16]]:
                        MRoPEQKNormPattern(
                            head_dim=layer.head_size,
                            num_heads=layer.num_heads,
                            num_kv_heads=layer.num_kv_heads,
                            eps=epsilon,
                            is_neox=neox,
                            mrope_section=mrope_section,
                            mrope_interleaved=interleaved,
                        ).register(self.patterns)

        self.dump_patterns(config, self.patterns)
        logger.info("MRoPE + QK norm fusion pass initialized with op: %s", fused_op)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused MRoPE+QK Norm on %s sites", self.matched_count)

    def uuid(self):
        return VllmInductorPass.hash_source(self, MRoPEQKNormPattern)
