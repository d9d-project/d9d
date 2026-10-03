# Aggregation Metrics

## About

d9d includes basic aggregation metrics that compute global sums and weighted means across all ranks.

## Usage

```python
import torch
from d9d.metric.impl.aggregation import SumMetric, WeightedMeanMetric

loss = WeightedMeanMetric()
loss.update(values=torch.tensor([0.5, 0.7]), weights=torch.tensor([32.0, 16.0]))

total_samples = SumMetric()
total_samples.update(torch.tensor(48.0))
```

## API Reference

::: d9d.metric.impl.aggregation.SumMetric

::: d9d.metric.impl.aggregation.WeightedMeanMetric
