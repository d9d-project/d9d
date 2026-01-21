from typing import Any

import torch
import torch.distributed as dist

from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric


class WeightedMeanMetric(Metric[torch.Tensor]):
    """
    Computes the weighted mean of values.

    Tracks the sum of weighted values and the sum of weights.
    """

    def __init__(self):
        """Constructs a WeightedMeanMetric object."""

        super().__init__()
        self._value = torch.scalar_tensor(0, dtype=torch.float32)
        self._weight = torch.scalar_tensor(0, dtype=torch.float32)

        self._is_synced = False
        self._synced_value = torch.scalar_tensor(0, dtype=torch.float32)
        self._synced_weight = torch.scalar_tensor(0, dtype=torch.float32)

        self._handles: list[dist.Work] | None = None

    def update(self, values: torch.Tensor, weights: torch.Tensor):
        self._value += (values * weights).sum()
        self._weight += weights.sum()

        self._is_synced = False

    def trigger_sync(self, dist_context: DistributedContext):
        self._synced_value = self._value.clone()
        self._synced_weight = self._weight.clone()
        self._is_synced = True

        self._handles = [
            dist.all_reduce(self._synced_value, op=dist.ReduceOp.SUM, async_op=True),
            dist.all_reduce(self._synced_weight, op=dist.ReduceOp.SUM, async_op=True)
        ]

    def wait_sync(self, dist_context: DistributedContext):
        if self._handles is None:
            raise RuntimeError("Sync was not triggered before")

        for handle in self._handles:
            handle.wait()
        self._handles = None

    def compute(self) -> torch.Tensor:
        if self._is_synced:
            return self._synced_value / self._synced_weight
        else:
            return self._value / self._weight

    def reset(self):
        self._value.fill_(0)
        self._weight.fill_(0)
        self._is_synced = False
        self._handles = None

    def to(self, device: str | torch.device | int):
        self._weight = self._weight.to(device)
        self._value = self._value.to(device)
        self._synced_weight = self._synced_weight.to(device)
        self._synced_value = self._synced_value.to(device)

    @property
    def accumulated_weight(self) -> torch.Tensor:
        """
        Returns the total weight accumulated so far.

        Returns:
            Scalar tensor with total weight.
        """

        if self._is_synced:
            return self._synced_weight

        return self._weight

    def state_dict(self) -> dict[str, Any]:
        return {
            "value": self._value,
            "weight": self._weight
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._value = state_dict["value"]
        self._weight = state_dict["weight"]
