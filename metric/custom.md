---
title: Custom Metrics
---

# Creating Custom Metrics

The `d9d` framework allows you to implement custom metrics by adhering to the `Metric` interface.

## Design Guidelines

Metric implementations usually follow this design:

*   **GPU Residency**: Metrics accumulate data directly on the GPU tensors to avoid CPU-GPU synchronization.
*   **Linearly Additive States**: Instead of storing unstable averages (e.g., "Current Accuracy"), store raw accumulation counts like "Total Correct" and "Total Samples". These are mathematically safe to sum via `all_reduce`. 

### Helper Components

We provide the `d9d.metric.component` package to simplify implementation:

*   **MetricAccumulator**: A helper object that handles the boilerplate of maintaining Local vs Synchronized versions of a metric state tensor. It supports standard reduction operations like Sum, Max, and Min.

## Examples

### MaxMetric

Below is an example of a `MaxMetric` that tracks the maximum value seen across all ranks using the `MetricAccumulator` helper.

```python
import torch
import torch.distributed as dist
from typing import Any

from d9d.metric import Metric
from d9d.metric.component import MetricAccumulator, MetricReduceOp
from d9d.core.dist_context import DistributedContext

class MaxMetric(Metric[torch.Tensor]):
    def __init__(self):
        # Initialize accumulator with -inf
        self._max_val = MetricAccumulator(
            torch.tensor(float('-inf')), 
            reduce_op=MetricReduceOp.max
        )

    def update(self, value: torch.Tensor):
        # Update local max (No communication)
        self._max_val.update(value)

    def sync(self, dist_context: DistributedContext):
        # Perform all_reduce across the world
        self._max_val.sync()

    def compute(self) -> torch.Tensor:
        # Return the synchronized value
        return self._max_val.value

    def reset(self):
        self._max_val.reset()
        
    def to(self, device: str | torch.device | int):
        self._max_val.to(device)

    # Stateful Protocol for Checkpointing
    def state_dict(self) -> dict[str, Any]:
        return {'max_val': self._max_val.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._max_val.load_state_dict(state_dict['max_val'])
```

::: d9d.metric.component
    options:
        show_root_heading: true
        show_root_full_path: true
