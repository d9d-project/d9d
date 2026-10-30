from typing import Any

import torch

from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric
from d9d.metric.component import MetricAccumulator


class SumMetric(Metric[torch.Tensor]):
    """Metric for the sum of input values."""

    def __init__(self):
        """Constructs the ``SumMetric`` object."""
        self._accumulator = MetricAccumulator(torch.scalar_tensor(0, dtype=torch.float32))

    def update(self, value: torch.Tensor):
        """Adds the sum of all elements of ``value`` to the metric.

        Args:
            value: The values to add.
        """
        self._accumulator.update(value.sum())

    def sync(self, dist_context: DistributedContext):
        self._accumulator.sync()

    def compute(self) -> torch.Tensor:
        return self._accumulator.value

    def reset(self):
        self._accumulator.reset()

    def to(self, device: str | torch.device | int):
        self._accumulator.to(device)

    def state_dict(self) -> dict[str, Any]:
        return {"accumulator": self._accumulator.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._accumulator.load_state_dict(state_dict["accumulator"])
