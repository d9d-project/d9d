import abc
from typing import Any, Generic, TypeVar

import torch
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext
from d9d.core.types import TensorTree

TComputeResult = TypeVar("TComputeResult", bound=TensorTree)


class Metric(abc.ABC, Stateful, Generic[TComputeResult]):
    """Base class for all metrics.

    A metric tracks statistics over time, e.g. during training, and can be synchronized across distributed
    processes. It supports checkpointing through the ``Stateful`` interface.
    """

    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """Updates the metric state with a new batch of data.

        Args:
            *args: Positional arguments required by the specific metric implementation.
            **kwargs: Keyword arguments required by the specific metric implementation.
        """

    @abc.abstractmethod
    def sync(self, dist_context: DistributedContext):
        """Synchronizes the metric state across distributed processes.

        It aggregates statistics from all ranks, e.g. with an all-reduce, so that every rank has the global state.

        Args:
            dist_context: The distributed context.
        """

    @abc.abstractmethod
    def compute(self) -> TComputeResult:
        """Computes the current value of the metric.

        Returns:
            The metric value: a single tensor or a PyTree of tensors, as declared by ``TComputeResult``.
        """

    @abc.abstractmethod
    def reset(self):
        """Resets the metric state to its initial values."""

    def to(self, device: str | torch.device | int):
        """Moves the metric state to a device.

        Args:
            device: The device to move the metric state to.
        """
