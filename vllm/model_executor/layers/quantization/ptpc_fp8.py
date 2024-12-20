from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.fp8 import cutlass_fp8_supported
from vllm.model_executor.layers.quantization.fbgemm_fp8 import (
    FBGEMMFp8Config, FBGEMMFp8LinearMethod)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear, normalize_e4m3fn_to_e4m3fnuz)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class PTPCFp8Config(FBGEMMFp8Config):
    """Config class for Per-Token-Per-Channel Fp8."""

    def __init__(self, ignore_list: Optional[List[str]] = None):
        super().__init__(ignore_list, 1.0) # Dummy values

    @classmethod
    def get_name(cls) -> str:
        return "ptpc_fp8"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PTPCFp8Config":
        ignore_list = cls.get_from_keys(config, ["modules_to_not_convert"])
        input_scale_ub = cls.get_from_keys(config, ["activation_scale_ub"])
        return cls(ignore_list=ignore_list, input_scale_ub=input_scale_ub)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignore_list):
                return UnquantizedLinearMethod()
            return FBGEMMFp8LinearMethod(self)
        return None
