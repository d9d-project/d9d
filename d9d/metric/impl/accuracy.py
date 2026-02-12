from typing import Any

import torch

from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric
from d9d.metric.component import MetricAccumulator


class BinaryAccuracyMetric(Metric[torch.Tensor]):
    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold

        self._correct = MetricAccumulator(torch.scalar_tensor(0, dtype=torch.long))
        self._total = MetricAccumulator(torch.scalar_tensor(0, dtype=torch.long))

    def update(self, probs: torch.Tensor, labels: torch.Tensor):
        probs = probs.reshape(-1)
        labels = labels.reshape(-1)

        if probs.numel() != labels.numel():
            raise ValueError("Predictions and labels should have the same number of elements")

        probs_bin = (probs >= self._threshold).to(labels.dtype)

        self._correct.update((probs_bin == labels).sum())
        self._total.update(labels.numel())

    def sync(self, dist_context: DistributedContext):
        self._correct.sync()
        self._total.sync()

    def compute(self) -> torch.Tensor:
        return self._correct.value / self._total.value.clamp(min=1)

    def reset(self):
        self._correct.reset()
        self._total.reset()

    def to(self, device: str | torch.device | int):
        self._correct.to(device)
        self._total.to(device)

    def state_dict(self) -> dict[str, Any]:
        return {"correct": self._correct.state_dict(), "total": self._total.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._correct.load_state_dict(state_dict["correct"])
        self._total.load_state_dict(state_dict["total"])
