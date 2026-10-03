# Hidden States Aggregation

## About

The `d9d.module.block.hidden_states_aggregator` package collects and reduces model hidden states during execution. Use it to keep hidden states for later use, such as reward modeling, custom distillation objectives or analysis. A reducing aggregator does not keep the full hidden states in memory.

Create an aggregator with the `create_hidden_states_aggregator` factory.

## Snapshots

`pack_with_snapshot` returns the collected states after an earlier "snapshot" tensor. The snapshot can hold data from earlier iterations or pipeline stages. This lets you build up states over iterative loops.

## Modes

### `HiddenStatesAggregationMode.no`

Collects nothing. `pack_with_snapshot` always returns `None`.

### `HiddenStatesAggregationMode.mean`

Reduces the hidden states as soon as it receives them. It needs an aggregation mask with shape `(batch, seq_len)`. For each call to `add_hidden_states`, it:

1.  Computes the masked mean of the hidden states over the sequence.
2.  Stores only the result with shape `(batch, hidden_size)`, not the full `(batch, seq_len, hidden_size)` tensor.

This reduces memory use when you collect states over many iterations.

## Usage

```python
import torch

from d9d.module.block.hidden_states_aggregator import (
    HiddenStatesAggregationMode,
    create_hidden_states_aggregator,
)

attention_mask = torch.ones(2, 16)
aggregator = create_hidden_states_aggregator(HiddenStatesAggregationMode.mean, agg_mask=attention_mask)

for _ in range(3):
    hidden_states = torch.randn(2, 16, 2048)
    aggregator.add_hidden_states(hidden_states)

snapshot = aggregator.pack_with_snapshot(None)  # (3, 2, 2048)
```

## API Reference

::: d9d.module.block.hidden_states_aggregator
