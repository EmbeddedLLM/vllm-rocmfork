from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config, Fp8LinearMethod, Fp8KVCacheMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.platforms import current_platform
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear)

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
            raise ValueError("ptpc_fpp8 quantization is supported only on ROCm")
        super().__init__(
            is_checkpoint_fp8_serialized=False,
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


class PTPCFp8LinearMethod(Fp8LinearMethod):
    """Linear method for Per-Token and Per-Channel FP8 Quantization.
    Only supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support float8_e4m3fn data type due to the limitation of
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
            layer.weight,
            scale=None,
            use_per_token_if_dynamic=True)
        
        # Update the layer with the new values.
        layer.weight = Parameter(qweight.t(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        layer.input_scale = None
        
        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer)
            # Activations not quantized for marlin.
            del layer.input_scale

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)
    
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=None,
            input_scale_ub=None,
            bias=bias,
            cutlass_fp8_supported=None,
            use_per_token_if_dynamic=True)
