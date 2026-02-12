import abc
import dataclasses
import typing
from typing import Protocol

from torch.optim import Optimizer

from d9d.core.dist_context import DistributedContext
from d9d.core.protocol import LRSchedulerProtocol


@dataclasses.dataclass(kw_only=True)
class InitializeLRSchedulerContext:
    """
    Context data required to initialize an LR scheduler.

    Attributes:
        dist_context: The distributed context.
        total_steps: The total number of training steps.
        optimizer: The optimizer instance that the scheduler will control.
    """

    dist_context: DistributedContext
    total_steps: int
    optimizer: Optimizer


@typing.runtime_checkable
class LRSchedulerProvider(Protocol):
    """
    Protocol for defining how Learning Rate schedulers are created.
    """

    @abc.abstractmethod
    def __call__(self, context: InitializeLRSchedulerContext) -> LRSchedulerProtocol:
        """
        Initializes the LR scheduler for a specific model pipeline stage.

        Args:
            context: Context for this operation.

        Returns:
            The instantiated LR scheduler adhering to the protocol.
        """
