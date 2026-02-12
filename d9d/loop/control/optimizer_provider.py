import abc
import dataclasses
import typing
from typing import Protocol

from torch import nn
from torch.optim import Optimizer

from d9d.core.dist_context import DistributedContext


@dataclasses.dataclass(kw_only=True)
class InitializeOptimizerStageContext:
    """
    Context data required to initialize an optimizer.

    Attributes:
        dist_context: The distributed context.
        model: The model instance for which parameters will be optimized.
    """

    dist_context: DistributedContext
    model: nn.Module


@typing.runtime_checkable
class OptimizerProvider(Protocol):
    """
    Protocol for defining how optimizers are created for model pipeline stages.
    """

    @abc.abstractmethod
    def __call__(self, context: InitializeOptimizerStageContext) -> Optimizer:
        """
        Initializes the optimizer for a specific training stage.

        Args:
            context: Context for this operation.

        Returns:
            The instantiated PyTorch optimizer.
        """
