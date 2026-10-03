# Metrics Overview

## About

The `d9d.metric` package provides one interface for tracking, accumulating and synchronizing statistics, such as accuracy, in distributed training.

## The Single-GPU Trap

Many practitioners come from single-GPU training or data science. They are used to CPU-based workflows with libraries such as `scikit-learn`:

```python
# Typical single-node pattern.
loss_val = loss_fn(pred, target).item()  # <--- Host sync point 1
history.append(loss_val)
# ... later ...
avg = np.mean(history)  # <--- Host sync point 2
sklearn.metrics.f1_score(all_preds, all_targets)
```

In large-scale distributed training, this approach fails:

*   **Pipeline stalls**: `.item()` and `.cpu()` make the host wait until the GPU finishes its queued work. Meanwhile the host cannot queue new work, so the GPU idles.
*   **Out-of-memory errors**: A Python list of predictions from many steps quickly fills host RAM.
*   **Partial view**: Rank 0 sees only its own data shard, so a loss logged from rank 0 alone is misleading.

## The d9d Approach

The `Metric` interface is:

*   **Distributed**: Each metric synchronizes its state across all ranks in its `sync` method.
*   **Async-compatible**: A `Metric` implementation can stay simple and synchronous. The training loop drives it through the [`AsyncMetricCollector`](../internals/metric_collector.md), which runs synchronization and computation on a side CUDA stream. The main training loop continues meanwhile.
*   **Stateful**: Metrics implement the `torch.distributed.checkpoint.stateful.Stateful` interface, so they are saved in checkpoints.
*   **Small**: `Metric` is a lightweight interface with no hidden state accounting. Implement its methods and respect the lifecycle below.

## The Metric Lifecycle

A metric in d9d follows this lifecycle:

1.  **Update**: Runs every training step. The metric accumulates data locally on the GPU, e.g. with `.add_()`. No communication happens.
2.  **Sync**: Runs at the logging interval. The metric aggregates data across all ranks, e.g. with `all_reduce`.
3.  **Compute**: Computes the final value from the synchronized data, e.g. total loss divided by total samples.
4.  **Reset**: Clears the state for the next logging window.

## Usage

### With the Trainer

Usually, you create and update metrics in your `TrainTask`. See the examples in the [Interfaces](../loop/interfaces/index.md) documentation.

### Manual Usage

You can also use d9d metrics without the `Trainer`. Called directly, `sync()` runs the reduction on the current CUDA stream. To overlap it with other work, call it within `torch.cuda.stream(...)`.

```python
from d9d.metric.impl.aggregation import WeightedMeanMetric

# 1. Create the metric.
metric = WeightedMeanMetric()
metric.to("cuda")

dataloader = ...
dist_context = ...  # The DistributedContext of the job.

# 2. Training loop.
for batch in dataloader:
    # ... forward, backward ...
    loss = ...
    num_tokens = ...

    # Update the local state (no communication).
    metric.update(values=loss, weights=num_tokens)

# 3. Synchronize and compute.
# Reduces the state across all ranks on the current stream.
metric.sync(dist_context)
print(f"Global average loss: {metric.compute()}")

# 4. Reset for the next epoch.
metric.reset()
```

## API Reference

::: d9d.metric
