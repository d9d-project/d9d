# Inference Loop

## About

The `d9d.loop` package also provides the execution engine for distributed inference. Like the `Trainer`, the `Inference` engine separates the *definition* of a job from its *execution*. It runs forward passes only.

## Configuration and Construction

You build the `Inference` engine with the `InferenceConfigurator`. It combines the infrastructure configuration, the [job configuration](./config.md) and the user logic into an `Inference` object that is ready to run.

### The Configuration Lifecycle

`InferenceConfigurator.configure()` runs a setup sequence similar to training, but for forward-only execution:

1.  **Distributed Context Initialization**:
    *   Builds the global [DistributedContext](../core/dist_context.md).

2.  **Seeding**:
    *   Sets distributed seeds, e.g. for reproducible sampling or validation splits.

3.  **Task Instantiation**:
    *   Creates the `InferenceTask`. It defines how inputs are built and what happens to the outputs, e.g. writing them to a JSONL file.

4.  **Event Bus Initialization**:
    *   Creates the global `EventBus`. The model provider and the task use it to [register custom hooks](./interfaces/events.md).
    *   Triggers the `EVENT_INFERENCE_CONFIG_STARTED` event.

5.  **Data Stream Construction**:
    *   Calls the `DataProvider` to build the `MicrobatchPackStream`. The stream yields one pack (the microbatches of one step) per iteration.
    *   Triggers the `EVENT_INFERENCE_DATA_STREAM_READY` event.

6.  **Model Materialization**:
    *   The `ModelStageFactory` builds the model, as in [training](train.md#the-configuration-lifecycle).
    *   You can reuse the `ModelProvider` from training.
    *   The pipeline schedule is always `PipelineScheduleInferenceConfig`. `InferenceConfig` has no pipelining settings.
    *   Triggers the `EVENT_INFERENCE_MODEL_STAGES_READY` event.

7.  **State Assembly**:
    *   The components are packed into the `InferenceJobState`.
    *   The `Inference` engine is created with this state and returned.

## The Inference Lifecycle

`Inference.infer()` runs the lifecycle below.

### 1. Initialization and Recovery

Before the loop starts:

1.  **Mode Switching**:
    *   Enters `torch.inference_mode()`, which disables gradient tracking and saves memory.
    *   Switches all model modules to `.eval()` mode, which affects dropout, batch norm and similar layers.
2.  **State Loading**:
    *   If the save directory holds a checkpoint of this job, the `StateCheckpointer` loads it.
    *   This restores the model state, the `JobSchedule` and the data stream position, so an interrupted job resumes where it stopped.
3.  **Context Entry**:
    *   Enters the UI, garbage collector and profiler contexts.
4.  **Ready Hook Trigger**: Triggers the `EVENT_INFERENCE_READY` event.

### 2. The Step Loop

The loop runs until it reaches `JobSchedule.total_steps`, as in [training](train.md#2-the-step-loop). For every step:

1.  Triggers the `EVENT_INFERENCE_STEP_PRE` event.
2.  **Microbatch Execution**:
    *   The `DevicePackStream` hands out a **pack** of $N$ microbatches, already on the device, as in [training](train.md#data-prefetching).
    *   Triggers the `EVENT_INFERENCE_FORWARD_PRE` event.
    *   The `InferenceTask` maps each microbatch to model inputs. After the forward pass, it processes the outputs of each microbatch.
    *   Unlike training, **no backward pass** runs.
    *   Triggers the `EVENT_INFERENCE_FORWARD_POST` event.

3.  **Maintenance**:
    *   **GC**: `ManualGarbageCollector` runs periodically to keep peak memory in check.
    *   **Event-Based Logic**: Triggers the `EVENT_INFERENCE_STEP_POST` event.
    *   **Advance**: The `JobSchedule` increments the step counter.

4.  **Checkpointing**:
    *   If configured, the engine saves the *progress* of the job. A restarted job then skips the data it already processed.

### 3. Finalization

1.  **Task-specific**: Calls `InferenceTask.finalize()`. Use it e.g. to flush and close output files.
2.  **Event-specific**: Triggers the `EVENT_INFERENCE_FINISHED` event.

## Usage

Build the engine with an `InferenceConfigurator`, then call `infer()`:

```python
from d9d.loop.run import InferenceConfigurator

inference = InferenceConfigurator(
    mesh=mesh_params,                  # Physical cluster layout
    parameters=config,                 # InferenceConfig (schedule, checkpointing, ...)

    model_provider=...,                # The same provider as in training
    task_provider=...,                 # Inference-specific logic (e.g. generation)
    data_provider=...,                 # Validation or test data
).configure()

inference.infer()
```

## API Reference

::: d9d.loop.run.InferenceConfigurator

::: d9d.loop.run.Inference
