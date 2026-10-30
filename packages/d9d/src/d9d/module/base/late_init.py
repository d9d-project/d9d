import typing
from typing import Protocol


@typing.runtime_checkable
class ModuleLateInit(Protocol):
    """Protocol for modules that support late parameter initialization."""

    def reset_parameters(self):
        """Resets the module parameters (i.e. performs random initialization)."""
