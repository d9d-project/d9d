from typing import Protocol, runtime_checkable

from torch.distributed.checkpoint.stateful import Stateful


@runtime_checkable
class OptimizerProtocol(Protocol, Stateful):
    """
    Protocol defining an interface for standard PyTorch Optimizer object.

    This protocol ensures that the wrapped optimizer supports standard
    API and state checkpointing via the Stateful interface.
    """

    def step(self):
        """Performs a single optimization step."""

        ...

    def zero_grad(self):
        """Sets the gradients of all optimized tensors to zero."""

        ...


@runtime_checkable
class LRSchedulerProtocol(Protocol, Stateful):
    """
    Protocol defining an interface for a Learning Rate Scheduler.

    This protocol ensures that the wrapped scheduler supports stepping
    and state checkpointing via the Stateful interface.
    """

    def step(self):
        """Performs a single learning rate scheduling step."""

        ...
