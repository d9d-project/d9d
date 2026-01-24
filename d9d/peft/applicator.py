from torch import nn

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel

from .base import PeftMethod


def inject_peft_and_freeze(method: PeftMethod, module: nn.Module) -> ModelStateMapper:
    """
    Applies a PEFT method to a module, freezes non-trained parameters, and prepares state mapping.

    This function performs three main steps:

    1. Sets `requires_grad=False` for all parameters in the module.
    2. Calls the method's `inject` to modify the model structure.
    3. Sets `requires_grad=True` for the parameters returned by the injection result.

    Args:
        method: The PEFT method strategy to apply.
        module: The PyTorch module to modify.

    Returns:
        A ModelStateMapper capable of loading checkpoint weights into the modified structure.
    """

    for param in module.parameters():
        param.requires_grad = False

    result = method.inject(module)

    for param in result.parameters_to_train:
        param.requires_grad = True

    return ModelStateMapperParallel(result.load_state_mappers)


def merge_peft(method: PeftMethod, module: nn.Module):
    """
    Merges PEFT adaptations back into the base model weights.

    Args:
        method: The PEFT method strategy originally applied.
        module: The PyTorch module to merge.
    """

    method.merge(module)
