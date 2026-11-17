from typing import Protocol, TypeVar, cast

from torch import nn

TModule = TypeVar("TModule", bound=nn.Module)


class _MarkedWindow(Protocol):
    _d9d_window: bool


def mark_window(module: TModule) -> TModule:
    """Marks a module as the root of a parameter group, without changing its forward or parameter names.

    Call it while building the model. The group of a marked module excludes the parameters of marked descendants.
    The mark describes membership, not execution order: ``forward_steps()`` decides the order.

    Args:
        module: The module to mark.

    Returns:
        The same module, without wrappers or hooks.
    """
    cast(_MarkedWindow, module)._d9d_window = True  # ruff: ignore[private-member-access] - The registration API owns this metadata.
    return module
