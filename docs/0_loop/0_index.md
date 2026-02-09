---
title: Training Loop
---

# Training Loop

## Overview

The `d9d.loop` package provides the execution engine for distributed training.

The `d9d` Trainer separates the *definition* of the job (Models, Tasks, Data) from the *execution* of the job (Synchronization, Checkpointing, Profiling).

This allows the same code to run on a single GPU or a 1000-GPU Pipeline Parallel cluster without modifications.

## Configuration & Construction

To ensure reproducibility, the Trainer is not instantiated directly with loose objects. It is built using the `TrainingConfigurator` and the **dependency injection** pattern.

### TrainingConfigurator

This class binds the 

* **[Infrastructure Configuration](../core/dist_context.md)**, 
* **[Job Configuration](./config.md)**, 
* and **[User Logic](./interfaces.md)** (Providers) 

into a `Trainer` object with prepared `TrainJobState`.

```python
from d9d.loop.run import TrainingConfigurator

trainer = TrainingConfigurator(
    mesh=mesh_params,                  # Physical cluster layout
    parameters=config,                 # Logic configuration (batch size, etc)
    
    # --- User Logic ---
    model_provider=...,                # How to build the model
    task_provider=...,                 # How to compute loss
    data_provider=...,                 # How to load data
    optimizer_provider=...,            # How to optimize
    lr_scheduler_provider=...          # LR scheduler
).configure()
```

## The Configuration Lifecycle

The `TrainingConfigurator.configure()` method does:

1.  **Distributed Context Initialization**:
    *   Constructs the global [DistributedContext](../core/dist_context.md), therefore initializing all the required NCCL process groups and `DeviceMesh`es.

2.  **Seeding**:
    *   Sets distributed seeds using the configured `base_seed`. This ensures model initialization and other initial states are deterministic. [More info](../internals/determinism.md).

3.  **Task Instantiation**:
    *   Instantiates the `TrainTask` object using specified `TrainTaskProvider`.

4.  **Data Loader Construction**:
    *   Calls the `DatasetProvider` to get the dataset and wraps it into a `DataLoader`. 
    *   The DataLoader will move all the Tensor data to this worker's device **automatically**.

5.  **Model Materialization**:
    *   The `ModelStageFactory` runs. This is the heavy lifting of initialization:
        1.  **Meta Init**: `ModelProvider` creates the model on the `meta` device (no memory usage).
        2.  **Parallelization**: `ModelProvider` applies `DTensor` sharding/replication to parameters.
        3.  **Materialization**: Empty tensors are allocated on the actual GPU.
        4.  **Wait**: Hard barrier to ensure all ranks allocated memory successfully.
        5.  **Parameter Reset**: `model.reset_parameters()` is called to generate random weights on GPU.
        6.  **Source Loading (Optional)**: If configured, a pretrained checkpoint (e.g., from HF) is streamed into the model using `ModelStateMapper`.

6.  **Optimizer and LR Scheduler Setup**:
    *   `OptimizerFactory` iterates over the model parameters.
    *   Calls `OptimizerProvider` and `LRSchedulerProvider`.

7.  **State Assembly**:
    *   All components (including internal ones) are packed into the `TrainJobState`.
    *   The `Trainer` is instantiated with this state and returned.

## Training

To run a train job, just call the `.train()` method on a `Trainer` object that is returned by configuration process.

## The Training Lifecycle

The `Trainer.train()` method orchestrates the following lifecycle. It is critical to understand this flow when debugging distributed issues or checking for side effects.

### 1. Initialization & Recovery

Before the loop starts:

1.  **Global Synchronization**: The trainer waits for all ranks to come online (`barrier`).
2.  **State Loading**: The `StateCheckpointer` checks the filesystem.
    *   If a checkpoint exists, it loads it into all the `Stateful` objects inside its `_state`.
    *   If no checkpoint exists, it starts from the first step.
3.  **Context Entry**: The trainer enters several context managers:
    *   **UI**: Renders a progress bar.
    *   **Logging**: Initiates a new run in selected experiment tracker and dumps run hyperparameters there. [More info](../internals/tracker_integration.md).
    *   **Garbage Collector**: Disables automatic Python garbage collection.
    *   **Profiler**: Starts `torch.profiler` hooks. [More info](../internals/profiling.md).
    *   **Gradient Manager**: Sets up backward hooks for synchronizing gradient states by all-reduce.
    *   **Gradient Clipper**: Looks for model parameters which gradients will be registered for clipping.

### 2. The Step Loop

For every global step (`step`), the trainer performs the following actions in strict order:

1.  **Microbatch Execution**
    * The `DataLoader` yields a "Batch Group" containing $N$ microbatches (calculated automatically based on `BatchingConfig`). 
    * We delegate to the `TrainTask` for mapping data before feeding it into the model.
    * The gradients will be **accumulated locally** using either regular multiple forward-backward calls if pipeline parallelism is disabled, either using our internal [pipelining API](../internals/pipelining.md). We delegate to `TrainTask` to compute loss values between forward and backward passes.
    * Last gradient accumulation triggers all-reduce synchronization. Communications may start overlapping here.
    * We delegate to `TrainTask` to accumulate local metrics (e.g., token counts, accuracy) into the `Metric` state.

2.  **Metric Synchronization**
    *   **Metric Sync Trigger**: `JobLogger` triggers an async reduction of all metrics across the world. [More info](../metric/0_index.md).

3.  **Gradient Synchronization**
    *   **Wait & Scale**: The `GradientManager` waits for all backward hooks to finish. It synchronizes the *total weighted loss* across the world to determine the scaling factor, then divides all gradients by this factor (essential for correct averaging when batch sizes vary due to masking/packing). [More info](../internals/grad_sync.md).

4.  **Gradient Clipping**
    *   The `GradientClipper` calculates the global L2 norm of all parameters.
    *   If `max_norm` is set, gradients are modified in-place.
    *   The total norm is logged.
    *   [More info](../internals/grad_norm.md).

5.  **Optimization**
    *   **Step**: The `Optimizer` updates model parameters.
    *   **Schedule**: The `LRScheduler` updates the learning rate for the *next* step.
    *   **Zero Grad**: The `GradientManager` clears gradients for the next iteration.

6.  **Logging & Maintenance**
    *   **Log**: Metrics are finalized and written to the tracker.
    *   **GC**: `ManualGarbageCollector` runs if the current step matches the GC period.
    *   **Advance**: The `Stepper` increments the step count.

7.  **Checkpointing**
    *   If the current step matches `checkpointing.period_steps`, checkpointing is triggered. This acts as a global barrier.

### 3. Finalization

1.  **Task-specific**: We delegate to the `TrainTask` to do its specific finalization work.

## API Reference

::: d9d.loop.run.TrainingConfigurator

::: d9d.loop.run.Trainer
