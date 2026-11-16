from typing import Protocol, TypeVar, cast, runtime_checkable

from torch import nn

from .steps import ModuleSupportsResumableForward

TModule = TypeVar("TModule", bound=nn.Module)


@runtime_checkable
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


def parameter_group_roots(module: nn.Module) -> list[nn.Module]:
    """Returns the roots of the parameter groups of a stage: the stage root and its marked descendants.

    Args:
        module: The stage module.

    Returns:
        The group roots, in registration order, without duplicate modules.
    """
    if not isinstance(module, ModuleSupportsResumableForward):
        return [module]
    return [
        module,
        *(child for child in module.modules() if child is not module and isinstance(child, _MarkedParameterGroup)),
    ]


def group_modules(module: nn.Module, roots: set[nn.Module]) -> list[nn.Module]:
    """Returns the modules of a group, stopping at nested group roots.

    Args:
        module: The root of the group.
        roots: All group roots of the stage.

    Returns:
        The modules whose own parameters belong to this group, each once.
    """
    result: dict[nn.Module, None] = {}
    # A loop instead of a nested recursive function, which would hold the modules in a reference cycle.
    pending = [module]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result[current] = None
        pending.extend(child for child in reversed(list(current.children())) if child not in roots)
    return list(result)
