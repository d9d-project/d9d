from collections.abc import Mapping
from typing import Any

import torch

from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric


class ComposeMetric(Metric[dict[str, Any]]):
    """Container metric that groups named child metrics into one metric.

    Synchronization, computation, reset, device moves and checkpointing apply to all children. ``compute()``
    returns a dictionary with the result of each child. Update the children directly.
    """

    def __init__(self, children: Mapping[str, Metric]):
        """Constructs the ``ComposeMetric`` object.

        Args:
            children: The child metrics by name.
        """
        self._children = children

    def update(self, *args: Any, **kwargs: Any):
        """Always raises, because the children must be updated directly.

        Raises:
            ValueError: If called.
        """
        raise ValueError("Cannot update ComposeMetric directly. Update its children instead.")

    def __getitem__(self, item: str) -> Metric:
        """Returns the child metric with the given name."""
        return self._children[item]

    @property
    def children(self) -> Mapping[str, Metric]:
        """The child metrics by name."""
        return self._children

    def sync(self, dist_context: DistributedContext):
        for metric in self._children.values():
            metric.sync(dist_context)

    def compute(self) -> dict[str, Any]:
        return {metric_name: metric.compute() for metric_name, metric in self._children.items()}

    def reset(self):
        for metric in self._children.values():
            metric.reset()

    def to(self, device: str | torch.device | int):
        for metric in self._children.values():
            metric.to(device)

    def state_dict(self) -> dict[str, Any]:
        return {metric_name: metric.state_dict() for metric_name, metric in self._children.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for metric_name, metric in self._children.items():
            metric.load_state_dict(state_dict[metric_name])
