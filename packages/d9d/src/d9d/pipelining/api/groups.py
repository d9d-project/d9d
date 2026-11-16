from typing import Protocol, TypeVar, cast

from torch import nn

TModule = TypeVar("TModule", bound=nn.Module)


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
    cast(_MarkedWindow, module)._d9d_window = True  # ruff: ignore[private-member-access] - The registration API owns this metadata.
    return module
