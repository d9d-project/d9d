---
title: Inference Loop
---

# Inference Loop

## Overview

The `d9d.loop` package provides the execution engine not only for training but also for high-scale distributed inference.

The inference engine shares the same core philosophy as the Trainer: separating the *definition* of the job from the *execution*.

## Configuration & Construction

The inference environment is assembled using the `InferenceConfigurator`.

### InferenceConfigurator

This class binds the infrastructure and user logic into a ready-to-execute `Inference` object.

```python
from d9d.loop.run import InferenceConfigurator

inference = InferenceConfigurator(
    mesh=mesh_params,                  # Physical cluster layout
    parameters=config,                 # Logic configuration (batch size, etc)
    
    model_provider=...,                # Same provider used in training
    task_provider=...,                 # Inference-specific logic (e.g., generation)
    data_provider=...,                 # Validation/Test dataset
).configure()
```

## The Configuration Lifecycle

The `InferenceConfigurator.configure()` method performs a setup sequence similar to training, but optimized for forward-only execution:

1.  **Distributed Context Initialization**:
    *   Constructs the global [DistributedContext](../core/dist_context.md).

2.  **Seeding**:
    *   Sets distributed seeds. Determinism is crucial in inference for reproducible sampling or validation splits.

3.  **Task Instantiation**:
    *   Instantiates the `InferenceTask`. This defines how inputs are processed and what to do with the outputs (e.g., writing to a JSONL file).

4.  **Data Loader Construction**:
    *   Creates a distributed `DataLoader` that handles sharding the inference dataset across ranks.

5.  **Model Materialization**:
    *   The `ModelStageFactory` runs to build the model.
    *   **Note**: This reuses the exact same `ModelProvider` as training.

6.  **State Assembly**:
    *   Components are packed into `InferenceJobState`.
    *   The `Inference` engine is instantiated.

## Inference Execution

To run the job, call the `.infer()` method on the configured object.

## The Inference Lifecycle

The `Inference.infer()` method orchestrates the execution flow. It is designed to be lean and memory-efficient.

### 1. Initialization & Recovery

Before the loop starts:

1.  **Mode Switching**:
    *   Enables `torch.inference_mode()`. This disables gradient calculation globally, saving significant memory.
    *   Sets all model modules to `.eval()` mode (affecting Dropout, BatchNorm, etc.).
2.  **State Loading**:
    *   The `StateCheckpointer` loads the model weights from the specified checkpoint.
    *   If the job was interrupted previously, it also restores the `Stepper` and `DataLoader` state to resume exactly where it left off.
3.  **Context Entry**:
    *   Enters UI, Garbage Collector, and Profiler contexts.

### 2. The Step Loop

For every step:

1.  **Microbatch Execution**:
    *   The `DataLoader` yields a batch group.
    *   The `InferenceTaskOperator` manages the execution. 
    *   Data is fed through the model.
    *   Unlike training, **no backward pass** is performed.

2.  **Maintenance**:
    *   **GC**: `ManualGarbageCollector` runs periodically to ensure peak memory usage is controlled.
    *   **Advance**: The `Stepper` increments.

3.  **Checkpointing**:
    *   If configured, the system saves the *progress* of the inference job. This allows restarting a long-running generation job on a massive dataset without re-processing the first half.

### 3. Finalization

1.  **Task-specific**: 
    *   Calls `InferenceTask.finalize()`. 
    *   This is typically used to close file handles (e.g., flushing the final lines of a generated dataset to disk).

## API Reference

::: d9d.loop.run.InferenceConfigurator

::: d9d.loop.run.Inference
