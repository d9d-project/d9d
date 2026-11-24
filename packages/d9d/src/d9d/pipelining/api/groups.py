from typing import Protocol, TypeVar, cast, runtime_checkable

from torch import nn

from .steps import ModuleSupportsResumableForward

TModule = TypeVar("TModule", bound=nn.Module)


@runtime_checkable
class _MarkedParameterGroup(Protocol):
    _d9d_parameter_group: bool


def mark_parameter_group(module: TModule) -> TModule:
    """Makes a module the root of its own parameter group.

    The group holds the parameters of the module and its descendants, except those of other marked modules. The
    parameters outside every marked module belong to the group of the stage root. The mark changes neither the forward
    nor the parameter names.

    Args:
        module: The module.

    Returns:
        The same module, so that the call can wrap an assignment.
    """
    # ruff: ignore[private-member-access] - mark_parameter_group() owns this attribute.
    cast(_MarkedParameterGroup, module)._d9d_parameter_group = True
    return module


def marked_parameter_groups(module: nn.Module) -> list[nn.Module]:
    """Returns the modules of a stage that ``mark_parameter_group()`` marked.

    Args:
        module: The stage root.

    Returns:
        The marked descendants, in the order of ``module.modules()``.

    Raises:
        ValueError: If the stage has marks but no resumable forward, so no region could request their groups.
    """
    marked: list[nn.Module] = [
        child for child in module.modules() if child is not module and isinstance(child, _MarkedParameterGroup)
    ]
    if marked and not isinstance(module, ModuleSupportsResumableForward):
        raise ValueError(
            f"The stage ({type(module).__name__}) marks parameter groups but has no forward_steps(). Implement "
            "ModuleSupportsResumableForward on the stage root."
        )
    return marked
