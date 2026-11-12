from typing import Protocol, TypeVar, cast, runtime_checkable

from torch import nn

from .steps import ModuleSupportsWindows

TModule = TypeVar("TModule", bound=nn.Module)


@runtime_checkable
class _MarkedWindow(Protocol):
    _d9d_window: bool


def mark_window(module: TModule) -> TModule:
    """Marks a module as a parameter group without changing its forward or parameter names.

    Mark children before execution. The containing group excludes parameters owned by marked descendants.
    Registration describes membership, not execution order. Boundaries in ``forward_steps()`` determine the order.

    Args:
        module: The root of the parameter group.

    Returns:
        The same module, without wrappers or execution hooks.
    """
    cast(_MarkedWindow, module)._d9d_window = True  # noqa: SLF001 - The registration API owns this metadata.
    return module


def window_groups(module: nn.Module) -> list[nn.Module]:
    """Returns the stage root and its explicitly marked parameter groups.

    Args:
        module: The stage module.

    Returns:
        The group roots, in registration order, without duplicate modules.
    """
    if not isinstance(module, ModuleSupportsWindows):
        return [module]
    return [module, *(child for child in module.modules() if child is not module and isinstance(child, _MarkedWindow))]


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
