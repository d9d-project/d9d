# Distributed Profiling

!!! warning "Internal API Warning"
    If you are utilizing the standard `d9d` training infrastructure, you **do not** need to call these functions manually. The framework automatically handles profiling based on configuration. This package is primarily intended for users extending `d9d`.

## About

The `d9d.internals.profiling` package provides distributed-aware wrappers around the standard PyTorch Profiler and the CUDA caching allocator memory snapshots.

In large-scale distributed training, profiling often becomes difficult due to:

1.  **File Naming**: Thousands of ranks writing to the same filename causes race conditions.
2.  **Storage Space**: Raw Chrome tracing JSON files can grow to gigabytes very quickly.
3.  **Synchronization**: Ensuring all ranks profile the same specific step without manual intervention.

The `Profiler` class solves these issues by automatically handling file naming based on the `DeviceMesh` coordinates, compressing traces into `.tar.gz` archives on the fly, and managing the profiling schedule (wait/warmup/active).

## Memory Snapshots

The `MemorySnapshotter` class records the CUDA caching allocator history (`torch.cuda.memory._record_memory_history`) and dumps it as a memory snapshot: every live allocation with its stack trace, plus the timeline of allocation and free events. `record_function` annotations - such as pipeline actions - are captured into the snapshot as well. Snapshots follow the same conventions as traces: they are named by the `DeviceMesh` coordinates, serialized as JSON and compressed into `.tar.gz` archives.

Every recording starts with a cleared history, so a snapshot only contains the events of its own recording. Allocations made before the recording started are still present in the snapshot, but without stack traces.

The standard `d9d` infrastructure takes two kinds of snapshots, both configured by `MemorySnapshotConfig`:

*   **Configuration**: records the whole job configuration (`TrainingConfigurator.configure()` or `InferenceConfigurator.configure()`) right after the `DistributedContext` is constructed. Saved into `<snapshots_dir>/configure/`.
*   **Steps**: records the first `active_steps` steps of every `period_steps`-long cycle of the global step. Cycles are aligned to the global step, so the very first step of the job - where optimizer states and gradient buffers are allocated - is always recorded. Saved into `<snapshots_dir>/step_<N>/`, where `N` is the number of steps completed by the end of the recording.

A recording interrupted by an exception is saved as well, so an out-of-memory failure leaves the snapshot of the allocations that caused it.

The size of a snapshot scales with `max_entries` (several KB per recorded event), so keep windows short on large models.

### Viewing

Convert a snapshot into a self-contained interactive HTML page:

```python
import json
import tarfile

from torch.cuda._memory_viz import trace_plot

with tarfile.open("step_1/rank-0-coord-0-0-0-0-0-0-memory.tar.gz") as tar:
    snapshot = json.load(tar.extractfile("rank-0-coord-0-0-0-0-0-0-memory.json"))

with open("memory.html", "w") as f:
    f.write(trace_plot(snapshot))
```

::: d9d.internals.profiling
