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

* **Distributed Aware**: Each metric knows how to synchronize its state across an ND-parallel environment via the `sync` method.
* **Async Compatible**: While `Metric` implementations themselves can remain simple and synchronous, they are designed to be driven by the [`AsyncMetricCollector`](../internals/metric_collector.md). This wrapper offloads the synchronization and computation to a side-stream, allowing the main training loop to continue while metrics are being reduced.
* **Stateful**: Metrics implement the `torch.distributed.checkpoint.stateful.Stateful` interface, allowing their state to be checkpointed seamlessly.
* **Clear**: Unlike some other libraries, d9d's `Metric` is a lightweight interface. It has no hidden state accounting or complex contracts. Just implement the interface and ensure you don't break the lifecycle.

### The Metric Lifecycle

A Metric in `d9d` follows a specific lifecycle:

1.  **Update**: Happens every train step. Data is aggregated locally on the GPU using methods like `.add_()`. No communication occurs here.
2.  **Sync**: Happens at the logging interval. The metric aggregates data across the world (e.g. `all_reduce`). 
3.  **Compute**: Calculates the final scalar (e.g., dividing total loss by total samples) using the synchronized data.
4.  **Reset**: Clears the internal state for the next logging window.

## Usage Examples

### Basic Usage

Typically, you want to just instantiate and update metrics within your `TrainTask` object.

See related examples in [Trainer](../0_loop/interfaces.md) documentation.

### Manual Usage

You may want to use d9d metrics manually, without using the Trainer object.

When used directly, the `sync()` method is blocking by default. You may call it within `torch.cuda.stream(...)` 
to overlap with computations.

```python
import torch
from d9d.metric.impl import WeightedMeanMetric
from d9d.core.dist_context import DistributedContext

# 1. Initialize
metric = WeightedMeanMetric()
metric.to("cuda")

dataloader = ...
dist_ctx = ...

# 2. Training Loop
for step, batch in enumerate(dataloader):
    # ... forward, backward ...
    loss = ... 
    num_tokens = ... 
    
    # Update local state (No communication, cheap)
    metric.update(values=loss, weights=num_tokens)

# 3. Synchronize & Compute
# This will block until all ranks finish all_reduce
metric.sync(dist_ctx)
print(f"Global Average Loss: {metric.compute()}")

# 4. Reset for next epoch
metric.reset()
```

::: d9d.metric
    options:
        show_root_heading: true
        show_root_full_path: true
