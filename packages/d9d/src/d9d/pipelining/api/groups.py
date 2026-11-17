from typing import Protocol, TypeVar, cast

from torch import nn

TModule = TypeVar("TModule", bound=nn.Module)


class _MarkedParameterGroup(Protocol):
    _d9d_parameter_group: bool


def mark_parameter_group(module: TModule) -> TModule:
    """Makes a module the root of its own parameter group.

    A resumable forward requests a group at each boundary, and the executor gathers the weights of that group for the
    region. The group holds the parameters of the module and its descendants, except those of marked descendants.
    The parameters outside every marked module belong to the group of the stage root. Mark the modules while
    building the model. The forward and the parameter names stay the same.

    Args:
        module: The module to mark.

    Returns:
        The same module, so that the call can wrap an assignment.
    """
    # ruff: ignore[private-member-access] - mark_parameter_group() owns this attribute.
    cast(_MarkedParameterGroup, module)._d9d_parameter_group = True
    return module
