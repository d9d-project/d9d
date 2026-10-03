# Container Metrics

## About

Managing many metrics one by one means repeating the sync, compute and reset calls for each of them. Container metrics bundle several metrics and manage them together.

## Compose Metric

`ComposeMetric` wraps a mapping from string keys to `Metric` instances into a single metric.

You cannot call `.update()` on a `ComposeMetric`, because its children can take different arguments. Update the children directly instead. The lifecycle methods `.sync()`, `.compute()`, `.reset()`, `.to()`, `.state_dict()` and `.load_state_dict()` apply to all children.

## Usage

```python
import torch
from d9d.metric.impl.aggregation import SumMetric, WeightedMeanMetric
from d9d.metric.impl.container import ComposeMetric

dist_context = ...  # The DistributedContext of the job.

# 1. Group metrics together.
metrics = ComposeMetric({
    "loss": WeightedMeanMetric(),
    "total_samples": SumMetric(),
})

# 2. Update each child with its own arguments.
metrics["loss"].update(torch.tensor(0.5), torch.tensor(32.0))
metrics["total_samples"].update(torch.tensor(32.0))

# 3. Lifecycle methods apply to all children.
metrics.to("cuda")
metrics.sync(dist_context)

# 4. compute() returns a dictionary from metric name to result.
results = metrics.compute()

# 5. Reset all metrics for the next epoch or evaluation.
metrics.reset()
```

## API Reference

::: d9d.metric.impl.container.ComposeMetric
