# Custom Metrics

## About

You can implement a custom metric by implementing the `Metric` interface. The `d9d.metric.component` package provides helpers for the distributed state.

## Design Guidelines

Metric implementations usually follow these rules:

*   **GPU residency**: Accumulate data in GPU tensors, so updates need no host sync.
*   **Additive states**: Do not store averages such as "current accuracy". Store raw counts such as "total correct" and "total samples". Counts stay correct when summed with `all_reduce`.

## Helper Components

`MetricAccumulator` keeps a local and a synchronized copy of a metric state tensor. It supports the `sum`, `max` and `min` reduction operations (`MetricReduceOp`). There is no average operation, because averages are not additive.

## Usage

This `MaxMetric` tracks the maximum value seen across all ranks with a `MetricAccumulator`.

```python
import torch
from typing import Any

from d9d.metric import Metric
from d9d.metric.component import MetricAccumulator, MetricReduceOp
from d9d.core.dist_context import DistributedContext

class MaxMetric(Metric[torch.Tensor]):
    def __init__(self):
        self._max_val = MetricAccumulator(
            torch.tensor(float('-inf')),
            reduce_op=MetricReduceOp.max
        )

    def update(self, value: torch.Tensor):
        # Updates the local maximum, no communication.
        self._max_val.update(value)

    def sync(self, dist_context: DistributedContext):
        # Runs all_reduce across the default process group.
        self._max_val.sync()

    def compute(self) -> torch.Tensor:
        # Returns the synchronized value.
        return self._max_val.value

    def reset(self):
        self._max_val.reset()

    def to(self, device: str | torch.device | int):
        self._max_val.to(device)

    # Stateful protocol for checkpointing.
    def state_dict(self) -> dict[str, Any]:
        return {'max_val': self._max_val.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._max_val.load_state_dict(state_dict['max_val'])
```

## API Reference

::: d9d.metric.component
