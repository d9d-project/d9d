---
title: Metrics
---

# Metrics

## About

The `d9d.metric` package provides a unified interface for tracking, accumulating, and synchronizing statistics (such as Accuracy) across a distributed environment.

## Why and How

### The Single-GPU Trap

Some practitioners coming from single-GPU training or standard data science backgrounds are used to workflows relying on good-old CPU-based libraries such as `scikit-learn`:

```python
# Typical single-node pattern
loss_val = loss_fn(pred, target).item() # <--- CPU Sync Point 1
history.append(loss_val)
# ... later ...
avg = np.mean(history)     # <--- CPU Sync Point 2
sklearn.metrics.f1_score(all_preds, all_targets)
```

In a large-scale distributed environment, this approach causes critical failures:

*   **Pipeline Stalls**: Calling `.item()` or `.cpu()` forces a synchronization that waits for the GPU to finish. This destroys the pipelining efficiency required for training large models.
*   **Out-of-Memory Errors**: Accumulating prediction history for many steps in a Python list will rapidly exhaust RAM.
*   **No Synchronization - Partial View**: Rank 0 only sees its own data shard. Logging loss from Rank 0 is misleading.

So, we have to do something with metric implementations to be performant and accurate.

### The d9d Solution

This package addresses issues described above by providing a `Metric` interface that is:

* **Distributed Aware**: Each metric is supposed to be able to synchronize its state across ND-parallel environment.
* **Async Compatible**: We separate the *triggering* of communication from the *waiting* for results. This allows the communication to happen in the background while the GPU continues computing the next micro-batch.
* **Stateful**: Metrics implement the `torch.distributed.checkpoint.stateful.Stateful` interface, allowing their state to be checkpointed seamlessly.
* **Clear**: Unlike some other distributed metric libraries such as `torchmetrics`, d9d's `Metric` interface is really just an interface. It has no state you have to account, no special contract you have to follow, nothing. Just implement the interface and do whatever you want the way you want, only make sure that you won't break the metric lifecycle.

### The Metric Lifecycle

A Metric in `d9d` follows a specific lifecycle:

1.  **Update**: Happens every train step. Data is aggregated locally on the GPU. No communication occurs here.
2.  **Trigger Sync**: Happens at the logging interval. The metric schedules asynchronous collective operations (like `all_reduce`) to aggregate data across the world.
3.  **Wait Sync**: Acts as a barrier. Ensures the collective ions from the previous step are finished.
4.  **Compute**: Calculates the final scalar (e.g., dividing total loss by total samples) using the synchronized data.
5.  **Reset**: Clears the internal state for the next logging window.

## Usage Examples

### Basic Usage

Typically, you want to just instantiate and update metrics within your Trainable object. 

See related examples in [Trainable](TODO) documentation.

### Implementing a Custom Metric

Metric implementations included in d9d usually follow this design:

*   **GPU Residency**: Metrics accumulate data directly on the GPU tensors, so no GPU-CPU synchronization involved.
*   **Linearly Additive States**: For instance, instead of storing the "Current Accuracy" (which is hard to average), we store "Total Correct" and "Total Samples". These values are mathematically safe to sum via `all_reduce`.

Below is an example of a `MaxMetric` that tracks the maximum value seen across all ranks (e.g., max GPU memory usage or max gradient norm).

```python
import torch
import torch.distributed as dist
from typing import Any

from d9d.metric import Metric
from d9d.core.dist_context import DistributedContext

class MaxMetric(Metric):
    def __init__(self):
        self._max_val = torch.tensor(float('-inf'), device='cuda')
        self._handle: dist.Work | None = None

    def update(self, value: torch.Tensor):
        # Keep local max
        self._max_val = torch.max(self._max_val, value)

    def trigger_sync(self, dist_context: DistributedContext):
        # Schedule async reduction across the world
        self._handle = dist.all_reduce(
            self._max_val, 
            op=dist.ReduceOp.MAX, 
            async_op=True
        )

    def wait_sync(self, dist_context: DistributedContext):
        self._handle.wait()
        self._handle = None

    def compute(self) -> torch.Tensor:
        return self._max_val

    def reset(self):
        self._max_val.fill_(float('-inf'))
        self._handle = None

    # Stateful Protocol for Checkpointing
    def state_dict(self) -> dict[str, Any]:
        return {'max_val': self._max_val}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._max_val = state_dict['max_val']
```


### Manual Usage

You may want to use d9d metrics manually, without using the Trainable object.

Using the built-in `WeightedMeanMetric`, which is commonly used for tracking Loss (weighted by batch size/token count).

```python
import torch
from d9d.metric.impl import WeightedMeanMetric
from d9d.core.dist_context import DistributedContext

# 1. Initialize
metric = WeightedMeanMetric()

dataloader = ...
dist_ctx = ...

# 2. Training Loop
for step, batch in enumerate(dataloader):
    # ... forward, backward ...
    loss = ... # scalar tensor
    num_tokens_in_loss = ... # scalar tensor
    
    # Update local state (No communication)
    metric.update(values=loss, weights=num_tokens_in_loss)

# 3. Synchronize
# Initiate communication across all GPUs
metric.trigger_sync(dist_ctx)

# Do other work here
# to hide communication latency.

# 4. Finalize and Print
# Block until communication is done
metric.wait_sync(dist_ctx)
print(f"Global Average Loss: {metric.compute()}")

# 5. Reset for next epoch
metric.reset()
```

::: d9d.metric
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.metric.impl
    options:
        show_root_heading: true
        show_root_full_path: true
