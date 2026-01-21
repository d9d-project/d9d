from collections.abc import Mapping
from typing import Any

import torch

from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric


class ComposeMetric(Metric[dict[str, Any]]):
    def __init__(self, children: Mapping[str, Metric]):
        self._children = children

    def update(self, *args: Any, **kwargs: Any):
        raise ValueError("Cannot update ComposeMetric directly - you can only update its children")

    def trigger_sync(self, dist_context: DistributedContext):
        for metric in self._children.values():
            metric.trigger_sync(dist_context)

    def wait_sync(self, dist_context: DistributedContext):
        for metric in self._children.values():
            metric.wait_sync(dist_context)

    def compute(self) -> dict[str, Any]:
        return {
            metric_name: metric.compute()
            for metric_name, metric in self._children.items()
        }

    def reset(self):
        for metric in self._children.values():
            metric.reset()

    def to(self, device: str | torch.device | int):
        for metric in self._children.values():
            metric.to(device)
