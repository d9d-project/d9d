from torch import nn

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperIdentity


def identity_mapper_from_module(module: nn.Module) -> ModelStateMapper:
    """
    Creates an identity mapper for every parameter in a single PyTorch module.

    It is useful when you want to define a "pass-through" pipeline where the
    source checkpoint keys are expected to exactly match the model's current
    parameter names (standard `load_state_dict` behavior).

    Args:
        module: The instantiated PyTorch model to inspect.
    """

    return ModelStateMapperParallel([ModelStateMapperIdentity(key) for key in module.state_dict()])
