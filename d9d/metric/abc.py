import abc
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext


class Metric(abc.ABC, Stateful):
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

        ...

    @abc.abstractmethod
    def trigger_sync(self, dist_context: DistributedContext):
        """
        Initiates the synchronization of the metric state across distributed processes.

        This method should start the collective operations (e.g., all-reduce) required
        to aggregate statistics, but should not block waiting for completion if possible.

        Args:
            dist_context: The distributed context.
        """

        ...

    @abc.abstractmethod
    def wait_sync(self, dist_context: DistributedContext):
        """
        Waits for the synchronization initiated by `trigger_sync` to complete.

        After this method returns, the metric state must be fully aggregated and
        consistent across ranks.

        Args:
            dist_context: The distributed context.
        """

        ...

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        """
        Computes the current value of the metric.

        Returns:
            The computed metric value.
        """

        ...

    @abc.abstractmethod
    def reset(self):
        """
        Resets the internal state of the metric to the initial values.
        """

        ...
