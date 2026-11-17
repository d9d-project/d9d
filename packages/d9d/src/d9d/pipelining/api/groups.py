from typing import Protocol, TypeVar, cast

from torch import nn

TModule = TypeVar("TModule", bound=nn.Module)


class _MarkedWindow(Protocol):
    _d9d_window: bool


def mark_window(module: TModule) -> TModule:
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
    # ruff: ignore[private-member-access] - mark_window() owns this attribute.
    cast(_MarkedWindow, module)._d9d_window = True
    return module
