# Metric Collection

## About

`AsyncMetricCollector` processes a metric without blocking the training loop. The [`Metric`](./overview.md) interface is synchronous. The collector wraps a metric and runs its synchronization and computation on a side CUDA stream. So the training loop continues at once and does not wait for the metric all-reduce. Collecting the results makes the current stream wait for the side stream, but not the host: the results stay tensors on the GPU.

The `Trainer` collects its metrics this way. You need the collector only in your own training loop or logging.

## Usage

```python
import torch.distributed as dist

from d9d.metric import AsyncMetricCollector
from d9d.metric.catalogue.aggregation import WeightedMeanMetric

metric = WeightedMeanMetric()
collector = AsyncMetricCollector(metric)
# Moves the metric to CUDA and creates the side stream.
collector.bind()

for step, batch in enumerate(dataloader):
    loss, num_tokens = ...
    metric.update(values=loss, weights=num_tokens)

    if step % 10 == 0:
        # Starts the all-reduce and the computation on the side stream.
        # Pass None in a single process to skip the synchronization.
        collector.schedule_collection(dist.group.WORLD)

        # ... other work of the step runs meanwhile ...

        # A GPU tensor. The collector also resets the metric.
        mean_loss = collector.collect_results()

collector.unbind()
```

## API Reference

::: d9d.metric.AsyncMetricCollector
