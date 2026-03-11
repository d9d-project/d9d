from typing import Any

import torch

from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric
from d9d.metric.component import MetricAccumulator


class WeightedMeanMetric(Metric[torch.Tensor]):
    """
    Computes the weighted mean of values.

    Tracks the sum of weighted values and the sum of weights.
    """

    def __init__(self):
        """Constructs a WeightedMeanMetric object."""

        self._value = MetricAccumulator(torch.scalar_tensor(0, dtype=torch.float32))
        self._weight = MetricAccumulator(torch.scalar_tensor(0, dtype=torch.float32))

    def update(self, values: torch.Tensor, weights: torch.Tensor):
        self._value.update((values * weights).sum())
        self._weight.update(weights.sum())

    def sync(self, dist_context: DistributedContext):
        self._value.sync()
        self._weight.sync()

    def compute(self) -> torch.Tensor:
        return self._value.value / self._weight.value

    def reset(self):
        self._value.reset()
        self._weight.reset()

    def to(self, device: str | torch.device | int):
        self._value.to(device)
        self._weight.to(device)

    @property
    def accumulated_weight(self) -> torch.Tensor:
        """
        Returns the total weight accumulated so far.

        Returns:
            Scalar tensor with total weight.
        """

        return self._weight.value

    def state_dict(self) -> dict[str, Any]:
        return {"value": self._value.state_dict(), "weight": self._weight.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._value.load_state_dict(state_dict["value"])
        self._weight.load_state_dict(state_dict["weight"])
