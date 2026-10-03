from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol for a standard PyTorch optimizer.

    A conforming optimizer supports stepping, zeroing gradients and checkpointing through the ``Stateful``
    interface (``state_dict`` and ``load_state_dict``).
    """

    def step(self):
        """Performs a single optimization step."""

    def zero_grad(self):
        """Sets the gradients of all optimized tensors to zero."""

    def state_dict(self) -> dict[str, Any]:
        """Returns the optimizer's state as a serializable dict.

        Returns:
            A dict containing the optimizer's state, suitable for checkpointing.
        """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the optimizer's state from a state dict.

        Args:
            state_dict: The state dict to restore from.
        """


@runtime_checkable
class LRSchedulerProtocol(Protocol):
    """Protocol for a learning rate scheduler.

    A conforming scheduler supports stepping and checkpointing through the ``Stateful`` interface
    (``state_dict`` and ``load_state_dict``).
    """

    def step(self):
        """Performs a single learning rate scheduling step."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Returns the scheduler's state as a serializable dict.

        Returns:
            A dict containing the scheduler's state, suitable for checkpointing.
        """
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the scheduler's state from a state dict.

        Args:
            state_dict: The state dict to restore from.
        """
