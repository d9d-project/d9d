import abc
from typing import Any, Generic, TypeVar

import torch
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext
from d9d.core.types import TensorTree

TComputeResult = TypeVar("TComputeResult", bound=TensorTree)


class Metric(abc.ABC, Stateful, Generic[TComputeResult]):
    """
    Abstract base class for all metrics.

    Metrics track statistics over time (e.g., during training) and can be synchronized
    across distributed processes. They also support state persistence via the Stateful
    interface.
    """

    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """
        Updates the metric state with a new batch of data.

        Args:
            *args: Positional arguments required by the specific metric implementation.
            **kwargs: Keyword arguments required by the specific metric implementation.
        """

    @abc.abstractmethod
    def sync(self, dist_context: DistributedContext):
        """
        Synchronizes the metric state across distributed processes.

        This method aggregates statistics from all ranks (e.g., via all-reduce)
        to ensure the metric state is consistent globally.

        Args:
            dist_context: The distributed context.
        """

    @abc.abstractmethod
    def compute(self) -> TComputeResult:
        """
        Computes the current value of the metric.

        Returns:
            The computed metric result (of type `TComputeResult`).
                This can be a single `torch.Tensor` or `PyTree` structure (dict, list, etc.)
                containing tensors, depending on how the subclass was typed.
        """

    @abc.abstractmethod
    def reset(self):
        """
        Resets the internal state of the metric to the initial values.
        """

    def to(self, device: str | torch.device | int):
        """
        Moves a metric state to a specified device.

        Args:
            device: The device to move the metric state to.
        """
