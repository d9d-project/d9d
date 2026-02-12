from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OptimizerProtocol(Protocol):
    """
    Protocol defining an interface for standard PyTorch Optimizer object.

    This protocol ensures that the wrapped optimizer supports standard
    API and state checkpointing via the Stateful interface.
    """

    def step(self):
        """Performs a single optimization step."""

    def zero_grad(self):
        """Sets the gradients of all optimized tensors to zero."""

    def state_dict(self) -> dict[str, Any]:
        """
        Objects should return their state_dict representation as a dictionary.
        The output of this function will be checkpointed, and later restored in
        `load_state_dict()`.

        Returns:
            Dict: The objects state dict
        """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore the object's state from the provided state_dict.

        Args:
            state_dict: The state dict to restore from
        """


@runtime_checkable
class LRSchedulerProtocol(Protocol):
    """
    Protocol defining an interface for a Learning Rate Scheduler.

    This protocol ensures that the wrapped scheduler supports stepping
    and state checkpointing via the Stateful interface.
    """

    def step(self):
        """Performs a single learning rate scheduling step."""

        ...

    def state_dict(self) -> dict[str, Any]:
        """
        Objects should return their state_dict representation as a dictionary.
        The output of this function will be checkpointed, and later restored in
        `load_state_dict()`.

        Returns:
            Dict: The objects state dict
        """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore the object's state from the provided state_dict.

        Args:
            state_dict: The state dict to restore from
        """
