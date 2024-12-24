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

import vllm._fp8gemm_C # noqa: F401
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


def apply_fp8_linear_custom(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = True,
    use_per_token_if_dynamic: bool = False,
) -> torch.Tensor:
    # ops.scaled_fp8_quant supports both dynamic and static quant.
    #   If dynamic, layer.input_scale is None and x_scale computed from x.
    #   If static, layer.input_scale is scalar and x_scale is input_scale.

    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    if out_dtype is None:
        out_dtype = input.dtype

    # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
    if cutlass_fp8_supported:
        qinput, x_scale = ops.scaled_fp8_quant(
            input_2d,
            input_scale,
            scale_ub=input_scale_ub,
            use_per_token_if_dynamic=use_per_token_if_dynamic)

        # Fused GEMM_DQ
        output = ops.cutlass_scaled_mm(qinput,
                                       weight,
                                       out_dtype=input.dtype,
                                       scale_a=x_scale,
                                       scale_b=weight_scale,
                                       bias=bias)
        return output.view(*output_shape)

    # torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    else:
        # Note: we pad the input because torch._scaled_mm is more performant
        # for matrices with batch dimension > 16.
        # This could change in the future.
        if input.dtype != torch.float8_e4m3fnuz:
            qinput, x_scale = ops.scaled_fp8_quant(
                input_2d,
                input_scale,
                num_token_padding=17,
                use_per_token_if_dynamic=use_per_token_if_dynamic)
        else:
            qinput, x_scale = input_2d, input_scale

        per_tensor_weights = (weight_scale.numel() == 1)
        per_tensor_activations = (x_scale.numel() == 1)

        if per_tensor_weights and per_tensor_activations:
            # Fused GEMM_DQ
            output = torch._scaled_mm(qinput,
                                      weight,
                                      out_dtype=out_dtype,
                                      scale_a=x_scale,
                                      scale_b=weight_scale,
                                      bias=bias)
            # A fix for discrepancy in scaled_mm which returns tuple
            # for torch < 2.5 and a single value in torch >= 2.5
            if type(output) is tuple and len(output) == 2:
                output = output[0]

            return torch.narrow(output, 0, 0,
                                input_2d.shape[0]).view(*output_shape)

        else:
            # Fallback for channelwise case, where we use unfused DQ
            # due to limitations with scaled_mm

            # Symmetric quantized GEMM by definition computes the following:
            #   C = (s_x * X) (s_w * W) + bias
            # This is equivalent to dequantizing the weights and activations
            # before applying a GEMM.
            #
            # In order to compute quantized operands, a quantized kernel
            # will rewrite the above like so:
            #   C = s_w * s_x * (X * W) + bias
            #
            # For the scaled_mm fallback case, we break this down, since it
            # does not support s_w being a vector.

            # Making sure the dummy tensor is on the same device as the weight
            # global TORCH_DEVICE_IDENTITY
            # if TORCH_DEVICE_IDENTITY.device != weight.device:
            #     TORCH_DEVICE_IDENTITY = TORCH_DEVICE_IDENTITY.to(weight.device)

            # GEMM
            # This computes C = (X * W).
            # Output in fp32 to allow subsequent ops to happen in-place
            # output = torch._scaled_mm(qinput,
            #                           weight,
            #                           scale_a=TORCH_DEVICE_IDENTITY,
            #                           scale_b=TORCH_DEVICE_IDENTITY,
            #                           out_dtype=torch.float32)
            # # A fix for discrepancy in scaled_mm which returns tuple
            # # for torch < 2.5 and a single value in torch >= 2.5
            # if type(output) is tuple and len(output) == 2:
            #     output = output[0]
            # # Unpad (undo num_token_padding)
            # output = torch.narrow(output, 0, 0, input_2d.shape[0])
            # x_scale = torch.narrow(x_scale, 0, 0, input_2d.shape[0])

            # # DQ
            # # C = sw * sx * (X * W) + bias
            # output = output * x_scale * weight_scale.t()
            # if bias is not None:
            #     output = output + bias
            output = torch.ops._fp8gemm_C.f8f8bf16_rowwise(qinput, weight, x_scale, weight_scale, bias, True)
            return output.to(dtype=input.dtype)


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
        layer.weight = Parameter(qweight, requires_grad=False)
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
    
        return apply_fp8_linear_custom(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=None,
            input_scale_ub=None,
            bias=bias,
            cutlass_fp8_supported=None,
            use_per_token_if_dynamic=True)
