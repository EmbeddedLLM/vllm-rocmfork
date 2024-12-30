from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
                                                         Fp8LinearMethod,
                                                         Fp8KVCacheMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.platforms import current_platform

import vllm._fp8gemm_C  # noqa: F401
from vllm import _custom_ops as ops

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class PTPCFp8Config(Fp8Config):
    """Config class for Per-Token-Per-Channel Fp8."""

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
    ) -> None:
        if not current_platform.is_rocm():
            raise ValueError("ptpc_fp8 quantization is supported only on ROCm")
        super().__init__(is_checkpoint_fp8_serialized=False,
                         activation_scheme=activation_scheme,
                         ignored_layers=ignored_layers)

    @classmethod
    def get_name(cls) -> str:
        return "ptpc_fp8"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PTPCFp8Config":
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        return cls(activation_scheme=activation_scheme,
                   ignored_layers=ignored_layers)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return PTPCFp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None


def apply_ptpc_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> torch.Tensor:
    # ops.scaled_fp8_quant supports both dynamic and static quant.
    #   If dynamic, layer.input_scale is None and x_scale computed from x.
    #   If static, layer.input_scale is scalar and x_scale is input_scale.

    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])

    if out_dtype is None:
        out_dtype = input.dtype

    if input.dtype != torch.float8_e4m3fnuz:

        qinput, x_scale = ops.scaled_fp8_quant(
            input_2d,
            input_scale,
            scale_ub=input_scale_ub,
            use_per_token_if_dynamic=use_per_token_if_dynamic)
    else:
        qinput, x_scale = input_2d, input_scale

    # At this stage, the qinput and weight should have been quantized as
    # per-token-activation
    # per-channel-weight

    output = torch.ops._fp8gemm_C.f8f8bf16_rowwise(qinput, weight, x_scale,
                                                   weight_scale, bias, True)
    return output.to(dtype=input.dtype)


class PTPCFp8LinearMethod(Fp8LinearMethod):
    """Linear method for Per-Token and Per-Channel FP8 Quantization.
    Only supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support float8_e4m3fnuz data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: PTPCFp8Config):
        super().__init__(quant_config=quant_config)
        # Force weight quantization
        self.quant_config.is_checkpoint_fp8_serialized = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)

        # Quantize the weights.
        qweight, weight_scale = ops.scaled_fp8_quant(
            layer.weight, scale=None, use_per_token_if_dynamic=True)

        # Update the layer with the new values.
        layer.weight = Parameter(qweight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        layer.input_scale = None

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        return apply_ptpc_fp8_linear(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=None,
                                     input_scale_ub=None,
                                     bias=bias,
                                     use_per_token_if_dynamic=True)
