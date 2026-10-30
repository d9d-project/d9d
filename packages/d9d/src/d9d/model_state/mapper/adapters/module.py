from torch import nn

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperIdentity


def identity_mapper_from_module(module: nn.Module) -> ModelStateMapper:
    """Creates an identity mapper for every key in the state dict of a PyTorch module.

    Use it when the checkpoint keys exactly match the module's state dict keys, as in standard
    ``load_state_dict`` behavior.

    Args:
        module: The instantiated PyTorch module to inspect.

    Returns:
        A composite identity mapper for the module's state.
    """
    return ModelStateMapperParallel([ModelStateMapperIdentity(key) for key in module.state_dict()])
