# Training Loop

## About

The `d9d.loop` package provides the execution engine for distributed training. The `Trainer` separates the *definition* of a job (models, tasks, data) from its *execution* (synchronization, checkpointing, profiling). So the same code runs on a single GPU or on a large pipeline-parallel cluster without changes.

## Configuration and Construction

You do not create a `Trainer` from loose objects. You build it with the `TrainingConfigurator`, using dependency injection.

The `TrainingConfigurator` combines:

*   the [infrastructure configuration](../core/dist_context.md),
*   the [job configuration](./config.md),
*   the [user logic](./interfaces/index.md) (providers).

It returns a `Trainer` that holds a prepared `TrainJobState`.

### The Configuration Lifecycle

`TrainingConfigurator.configure()` runs these steps:

1.  **Distributed Context Initialization**:
    *   Builds the global [DistributedContext](../core/dist_context.md). This creates the NCCL process groups and `DeviceMesh` objects.

2.  **Seeding**:
    *   Sets distributed seeds from the configured `base_seed`. Model initialization and other initial states are then deterministic. [More info](../internals/determinism.md).

3.  **Task Instantiation**:
    *   Creates the `TrainTask` with the `TrainTaskProvider`.

4.  **Event Bus Initialization**:
    *   Creates the global `EventBus`. The model provider and the task use it to [register custom hooks](./interfaces/events.md).
    *   Triggers the `EVENT_TRAIN_CONFIG_STARTED` event.

5.  **Data Stream Construction**:
    *   Calls the `DataProvider` to build the `MicrobatchPackStream`. The stream yields one pack (the microbatches of one step) per iteration.
    *   Triggers the `EVENT_TRAIN_DATA_STREAM_READY` event.

6.  **Model Materialization**:
    *   The `ModelStageFactory` builds each model stage:
        1.  **Meta Init**: The `ModelProvider` creates the model on the `meta` device, so no memory is allocated.
        2.  **Parallelization**: The `ModelProvider` shards or replicates the parameters as `DTensor`.
        3.  **Materialization**: Uninitialized tensors are allocated on the GPU.
        4.  **Parameter Reset**: `model.reset_parameters()` fills the weights with random values on the GPU.
        5.  **Source Loading (Optional)**: If configured, a pretrained checkpoint (e.g. from Hugging Face) is streamed into the model through its `ModelStateMapper`.
    *   Triggers the `EVENT_TRAIN_MODEL_STAGES_READY` event.

7.  **Optimizer and LR Scheduler Setup**:
    *   The `OptimizerFactory` calls the `OptimizerProvider` and the `LRSchedulerProvider` once per local model stage.
    *   Triggers the `EVENT_TRAIN_OPTIMIZER_READY` and `EVENT_TRAIN_LR_SCHEDULER_READY` events.

8.  **State Assembly**:
    *   All components, including internal ones, are packed into the `TrainJobState`.
    *   The `Trainer` is created with this state and returned.

## The Training Lifecycle

`Trainer.train()` runs the lifecycle below. Knowing this order helps when you debug distributed issues or look for side effects.

### 1. Initialization and Recovery

Before the loop starts:

1.  **Global Synchronization**: The trainer waits for all ranks (a barrier).
2.  **State Loading**: The `StateCheckpointer` looks for a checkpoint in the save directory.
    *   If a checkpoint exists, it loads it into all `Stateful` objects of the job state.
    *   If no checkpoint exists, the job starts from the first step.
3.  **Context Entry**: The trainer enters several context managers:
    *   **UI**: Shows a progress bar.
    *   **Logging**: Starts a new run in the experiment tracker and logs the run hyperparameters. [More info](../internals/tracker_integration.md).
    *   **Garbage Collector**: Disables automatic Python garbage collection.
    *   **Profiler**: Starts the `torch.profiler` hooks. [More info](../internals/profiling.md).
    *   **Gradient Manager**: Installs the backward hooks that all-reduce gradients.
    *   **Gradient Clipper**: Groups the parameters for global gradient norm computation.
    *   **Metric Collector**: Binds the async metric collector to the device.
4.  **Ready Hook Trigger**: Triggers the `EVENT_TRAIN_READY` event.

### 2. The Step Loop

The loop runs until it reaches `JobSchedule.total_steps` (see [Data Loading](./interfaces/data.md)). For every step, the trainer runs these actions in this order:

1.  Triggers the `EVENT_TRAIN_STEP_PRE` event.
2.  **Microbatch Execution**
    *   The `DevicePackStream` hands out a pack of $N$ microbatches, already on the device (see [Data Prefetching](#data-prefetching)).
    *   Triggers the `EVENT_TRAIN_FORWARD_BACKWARD_PRE` event.
    *   The `TrainTask` maps each microbatch to model inputs.
    *   The [pipeline program](../internals/pipelining.md) runs the forward and backward passes over all microbatches of the pack. Without pipeline parallelism, the program has a single stage.
    *   Gradients accumulate locally. Between the forward and backward passes, the `TrainTask` computes the loss.
    *   The last accumulation of each gradient bucket starts its all-reduce, which overlaps with the remaining backward work.
    *   The `TrainTask` updates the local metrics (e.g. token counts, accuracy) for each microbatch.
    *   Triggers the `EVENT_TRAIN_FORWARD_BACKWARD_POST` event.

3.  **Metric Synchronization**
    *   `JobLogger` starts an async reduction of all metrics across the ranks. [More info](../metric/overview.md).

4.  **Gradient Synchronization**
    *   **Wait & Scale**: The `GradientManager` waits for all backward hooks to finish. It then sums the loss weights across the ranks and divides all gradients by this total. This gives a correct average when the number of tokens varies, e.g. due to masking or packing. [More info](../internals/grad_sync.md).

5.  **Gradient Clipping**
    *   The `GradientClipper` computes the global L2 norm of all gradients.
    *   If `max_norm` is set, the gradients are clipped in place.
    *   The total norm is logged.
    *   [More info](../internals/grad_norm.md).

6.  **Optimization**
    *   Triggers the `EVENT_TRAIN_OPTIMIZER_STEP_PRE` event.
    *   **Step**: The optimizer updates the model parameters.
    *   Triggers the `EVENT_TRAIN_OPTIMIZER_STEP_POST` event.
    *   **Schedule**: The LR scheduler updates the learning rate for the *next* step.

7.  **Logging & Maintenance**
    *   **Log**: The loss and, on logging steps, the metrics are written to the tracker.
    *   **Zero Grad**: The `GradientManager` clears the gradients for the next step.
    *   **GC**: `ManualGarbageCollector` runs if the current step matches the GC period.
    *   **Event-Based Logic**: Triggers the `EVENT_TRAIN_STEP_POST` event.
    *   **Advance**: The `JobSchedule` increments the step counter.

8.  **Checkpointing**
    *   If the step matches `checkpointing.period_steps` or is the last step, the trainer saves a checkpoint. This is a global barrier.

### 3. Finalization

1.  **Task-specific**: The `TrainTask` runs its `finalize` method.
2.  **Event-specific**: Triggers the `EVENT_TRAIN_FINISHED` event.

## Data Prefetching

While a step runs, a background thread prepares the next `data_prefetch.prefetch_factor` packs. It loads and pins them, then copies them to the device on a side CUDA stream. All of this overlaps with compute.

*   **Background thread**: The data stream is iterated on this thread. With `num_workers: 0`, this includes the dataset and collator code.
*   **Pinned memory** makes the copies asynchronous. Set `pin_memory` in `AutoDataConfig`, or wrap a custom stream in `PinMemoryMicrobatchPackStream`. Without prefetching, pinning only adds a host-side copy.
*   **Memory**: Every prefetched pack occupies device memory. `prefetch_factor: 0` disables prefetching.
*   **Checkpoints** save the data position of the last completed step. Packs that were prefetched but not consumed are read again after a restart. A checkpoint can be loaded with any `prefetch_factor`.

## Sleep and Wake (Colocated RL)

In **colocated RL**, a rollout or inference engine shares the same GPUs as the `Trainer`. The two cannot occupy the device at once, so the `Trainer` must hand the GPU back for a while. `Trainer.sleep()` releases the training state to host memory and frees the device cache. `Trainer.wake()` restores it. Both are built on the [State Offloading](../core/offload.md) subsystem.

Both calls are collective: every rank must call them with the same tags. Call `sleep()` only between steps, e.g. from an `EVENT_TRAIN_STEP_POST` handler.

```python
trainer = TrainingConfigurator(...).configure()

# ... between training steps ...
trainer.sleep()                 # Offload the training state, free the GPU
rollout_engine.generate(...)    # Colocated inference runs on the freed device
trainer.wake()                  # Restore the training state, resume stepping
```

### What Gets Offloaded

`sleep()` selects subsystems with a [`SleepTag`](../core/offload.md#sleep-tags). The default, `SleepTag.TENSOR_STATES`, offloads all GPU tensor state as a single unit:

*   **Model** parameters and buffers (`TrackedModules`),
*   **Optimizer** state (`PipelinedOptimizer`),
*   **Gradient** buckets and the residual loss accumulator (`GradientManager`).

Data packs are not offloaded. The pack of the finished step and up to `prefetch_factor` [prefetched](#data-prefetching) packs stay on the device.

`SleepTag.COMMS` (NCCL process groups) is reserved but not implemented yet. Requesting it raises `NotImplementedError`.

The round trip keeps object identity, optimizer state dict keys and `DTensor` metadata. So external references, e.g. a frozen reference model held by a task, stay valid after waking. See the [round-trip invariant](../core/offload.md) for details.

### Order of Operations

`sleep()` runs, in order:

1.  Triggers `EVENT_TRAIN_SLEEP_PRE` (the state is still on the GPU), then a barrier.
2.  Offloads the gradient state, then the optimizer state, then the model parameters and buffers.
3.  A barrier, then triggers `EVENT_TRAIN_SLEEP_POST`.

`wake()` restores in the reverse order (model, optimizer, gradient state). Barriers and the `EVENT_TRAIN_WAKE_PRE` / `EVENT_TRAIN_WAKE_POST` events surround it.

## Usage

Build the trainer with a `TrainingConfigurator`, then call `train()`:

```python
from d9d.loop.run import TrainingConfigurator

trainer = TrainingConfigurator(
    mesh=mesh_params,                  # Physical cluster layout
    parameters=config,                 # TrainerConfig (schedule, checkpointing, ...)

    # --- User Logic ---
    model_provider=...,                # How to build the model
    task_provider=...,                 # How to compute the loss
    data_provider=...,                 # How to load data
    optimizer_provider=...,            # How to optimize
    lr_scheduler_provider=...,         # LR scheduler
).configure()

trainer.train()
```

## API Reference

::: d9d.loop.run.TrainingConfigurator

::: d9d.loop.run.Trainer
