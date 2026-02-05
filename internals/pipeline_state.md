---
title: Pipeline State Management
---

# Pipeline State Management

!!! warning "Warning:" 
    If you are utilizing the standard `d9d` training infrastructure, you **do not** need to manage pipeline states manually. The framework automatically handles this. This package is primarily intended for users extending `d9d`.

## About

The `d9d.internals.pipeline_state` package provides a unified mechanism to manage data lifecycle within a training step. It specifically addresses the complexity of transitioning between the **Global Context** (an entire training step/batch) and the **Sharded Context** (partial execution, i.e. within pipeline parallel loss computation).

For instance, a typical data flow in a pipelined step is:

1. Prepare the data using a **global** view.
2. Compute loss value for a microbatch, it now requires to create a **sharded** view of the data.
3. Log metrics, using a **global** view again.

`PipelineState` abstracts the slicing (Global -> Sharded) and aggregation (Sharded -> Global) operations behind a simple dictionary-like interface, allowing the training loop to act as a seamless bridge between these two contexts.


::: d9d.internals.pipeline_state
    options:
        show_root_heading: true
        show_root_full_path: true
