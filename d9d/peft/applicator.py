from torch import nn

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel

from .base import PeftMethod


def inject_peft_and_freeze(method: PeftMethod, module: nn.Module) -> ModelStateMapper:
    """Applies a PEFT method to a module and freezes all parameters it does not train.

    1.  Sets ``requires_grad=False`` for all parameters in the module.
    2.  Calls ``method.inject`` to modify the module structure.
    3.  Sets ``requires_grad=True`` for the parameters returned by the injection.

    Args:
        method: The PEFT method to apply.
        module: The PyTorch module to modify.

    Returns:
        A ``ModelStateMapper`` that loads checkpoint weights into the modified structure.
    """
    for param in module.parameters():
        param.requires_grad = False

    result = method.inject(module)

    for param in result.parameters_to_train:
        param.requires_grad = True

    return ModelStateMapperParallel(result.load_state_mappers)


def merge_peft(method: PeftMethod, module: nn.Module):
    """Merges PEFT adaptations back into the base model weights.

    Args:
        method: The PEFT method that was applied.
        module: The PyTorch module to merge.
    """
    method.merge(module)
