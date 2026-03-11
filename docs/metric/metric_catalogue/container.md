# Container Metrics

Managing multiple metrics individually can lead to boilerplate code, especially when manually synchronizing, computing, and resetting states across distributed workers. The `d9d` framework provides container metrics to bundle and manage collections of metrics efficiently.

## Compose Metric

The `ComposeMetric` wraps a mapping string keys to `Metric` instances into a single unifying interface. 

By design, you cannot call `.update()` directly on a `ComposeMetric` since different metrics may require different underlying arguments. Instead, you access and update the children directly. However, collective lifecycle methods like `.sync()`, `.compute()`, `.reset()`, `.state_dict()`, and `.to()` automatically cascade to all underlying metrics.

```python
import torch
from d9d.metric.impl.aggregation import SumMetric, WeightedMeanMetric
from d9d.metric.impl.container import ComposeMetric

# 1. Group multiple metrics together
metrics = ComposeMetric({
    "loss": WeightedMeanMetric(),
    "total_samples": SumMetric(),
})

# 2. Update children individually based on their specific signatures
metrics["loss"].update(torch.tensor(0.5), torch.tensor(32.0))
metrics["total_samples"].update(torch.tensor(32.0))

# 3. Lifecycle operations naturally propagate to all children via the container
metrics.to("cuda")
metrics.sync(dist_context)

# 4. Compute returns a dictionary mapping metric names to their final aggregated results
results = metrics.compute()

# 5. Reset all metrics for the next epoch/evaluation step
metrics.reset()
```

::: d9d.metric.impl.container.ComposeMetric
    options:
      heading_level: 3
