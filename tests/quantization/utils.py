from typing import Dict, List, Optional, Sequence, Tuple, Union
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.platforms import current_platform
import torch


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, all quantization methods require Nvidia or AMD GPUs
    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        return False

    capability = current_platform.get_device_capability()
    assert capability is not None

    min_capability = get_quantization_config(quant_method).get_min_capability()

    return capability.to_int() >= min_capability


TokensTextLogprobs = Tuple[
    List[int], str, Optional[Union[List[Dict[int, float]], SampleLogprobs]]
]

TextTextLogprobs = Tuple[
    List[str], str, Optional[Union[List[Dict[str, float]], List[Dict[str, Logprob]]]]
]

TokensTextLogprobsPromptLogprobs = Tuple[
    List[int],
    str,
    Optional[Union[List[Dict[int, float]], SampleLogprobs]],
    Optional[Union[List[Optional[Dict[int, float]]], PromptLogprobs]],
]

ModelOutputSequence = Sequence[
    Union[TokensTextLogprobs, TokensTextLogprobsPromptLogprobs, TextTextLogprobs]
]


def extract_log_probs(log_prob_dict_seq: Sequence[Dict[str, Logprob]]) -> List[float]:
    return [
        logprob.logprob
        for log_prob_dict in log_prob_dict_seq
        for logprob in log_prob_dict.values()
    ]


def extract_log_probs_from_model_ouput_sequence(
    model_output_sequence: ModelOutputSequence,
) -> torch.Tensor:
    log_probs_all = []
    for model_output in model_output_sequence:
        if len(model_output) == 3:
            _, _, logprobs_list = model_output
        elif len(model_output) == 4:
            _, _, logprobs_list, _ = model_output
        else:
            raise ValueError(
                f"Outputs tuple must have 3 or 4 elements but "
                f"{len(model_output)} elements were provided: "
                f"{model_output}"
            )

        log_probs_all.extend(extract_log_probs(logprobs_list))
    return torch.tensor(log_probs_all)


def check_target_closer(
    base_output_sequence: ModelOutputSequence,
    target_a_output_sequence: ModelOutputSequence,
    target_b_output_sequence: ModelOutputSequence,
) -> None:
    """Compare the logprobs of target outputs against a base model output
    to determine if target a is closer to the base model than target b.

    Args:
      base_output_sequence: Output from the base model
      target_a_out: Output from the first target model
      target_b_out: Output from the second target model
    """

    base_model_log_probs = extract_log_probs_from_model_ouput_sequence(
        base_output_sequence
    )
    target_1_log_probs = extract_log_probs_from_model_ouput_sequence(
        target_a_output_sequence
    )
    target_2_log_probs = extract_log_probs_from_model_ouput_sequence(
        target_b_output_sequence
    )
    assert torch.linalg.norm(
        base_model_log_probs - target_1_log_probs
    ) < torch.linalg.norm(base_model_log_probs - target_2_log_probs)


def check_logprobs_close(
    a_output_sequence: ModelOutputSequence,
    b_output_sequence: ModelOutputSequence,
    rtol: float,
    atol: float,
) -> None:
    """
    Compare log probabilities of two model output sequences for closeness.
    """

    a_log_probs = extract_log_probs_from_model_ouput_sequence(a_output_sequence)
    b_log_probs = extract_log_probs_from_model_ouput_sequence(b_output_sequence)
    assert torch.allclose(a_log_probs, b_log_probs, rtol=rtol, atol=atol)
