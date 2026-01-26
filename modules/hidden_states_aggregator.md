---
title: Hidden States Aggregation
---

# Hidden States Aggregation

## About

The `d9d.module.block.hidden_states_aggregator` package provides interfaces and implementations for collecting, reducing, and managing model hidden states during execution.

This is particularly useful in pipelines where intermediate activations need to be analyzed or stored (e.g., for reward modeling, custom distillation objectives, or analysis) without keeping the entire raw tensor history in memory.

As an end user, you typically will instantiate an aggregator with a factory method `create_hidden_states_aggregator`.

Aggregators support a `pack_with_snapshot` mechanism. This allows combining currently collected states with a pre-existing "snapshot" tensor (historical data or from previous pipeline stages), facilitating state management in stateful or iterative loops.


## Modes

### `HiddenStatesAggregationMode.noop`

Acts as a "null"-aggregator.

### `HiddenStatesAggregationMode.mean`
 The **Mean** mode (`HiddenStatesAggregationMode.mean`) performs "eager" reduction. Instead of storing the full `[Batch, Seq_Len, Hidden_Dim]` tensors for every step, it:
 
 1. Takes an aggregation mask.
 2. Computes the masked average immediately upon receiving the hidden states.
 3. Stores only the reduced `[Batch, Hidden_Dim]` vectors.
 
This significantly reduces memory footprint when accumulating states over many iterations.


::: d9d.module.block.hidden_states_aggregator
    options:
        show_root_heading: true
        show_root_full_path: true
