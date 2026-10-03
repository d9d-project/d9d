# State Offloading

## About

The `d9d.core.offload` package moves GPU-resident training state to host (CPU) memory and back. It is the base of the **sleep / wake** API of the `Trainer`. Sleep frees the GPUs for a colocated workload, for example a rollout engine that shares the GPUs in colocated reinforcement learning.

The package defines two things:

1.  **The `Offloadable` protocol**: the contract for a subsystem that owns GPU memory and can release and restore it.
2.  **The tensor primitives** `offload_tensor` and `onload_tensor`: they move the storage of one tensor to the host and back. The tensor object, and the `DTensor` wrapper, stay the same.

The user entry points `Trainer.sleep()`, `Trainer.wake()` and `Trainer.is_sleeping()` are documented on the [Training Loop](../loop/train.md) page. This page covers the primitives they are built on.

## The Round-Trip Guarantee

**An `offload` followed by an `onload` changes nothing observable.** Across the round trip:

*   Parameters and buffers keep their **object identity**.
*   The optimizer keeps its **state dict keys** and the tensor objects they map to.
*   `DTensor` **wrapper instances** keep their `device_mesh`, `placements`, global `shape`, `stride` and `dtype`.

Only the device storage is allocated again. So external references stay valid after wake-up. Gradient hooks, optimizer state keyed by parameter and a frozen reference model held by a task all point at the same objects.

The primitives swap the storage in place instead of creating new tensors. For a `DTensor`, only the storage of the local shard moves. The distributed metadata stays on the wrapper.

## Sleep Tags

`SleepTag` selects the subsystems that `Trainer.sleep` and `Trainer.wake` act on:

*   **`SleepTag.TENSOR_STATES`**: all GPU tensor state (model parameters and buffers, optimizer state, gradient buckets and the residual loss accumulator). They are offloaded together. `DEFAULT_SLEEP_TAGS` holds only this tag.
*   **`SleepTag.COMMS`**: NCCL process groups. Opt-in and **not implemented yet**. Requesting it raises `NotImplementedError`.

## Usage

### Offloading a Tensor

`offload_tensor` returns an `OffloadedTensor` handle. Keep it until you call `onload_tensor`, and pass the same tensor object to both calls.

```python
import torch
from d9d.core.offload import offload_tensor, onload_tensor

device = torch.device("cuda")
param = torch.randn(4096, 4096, device=device)

# Release the GPU storage; `param` now lives in host memory.
handle = offload_tensor(param, pin_memory=True)
assert param.device.type == "cpu"

# ... a colocated workload runs on the freed GPU ...

# Restore the GPU storage in place; `param` is the same object as before.
onload_tensor(param, handle, device=device)
assert param.device.type == "cuda"
```

For a `DTensor`, pass the wrapper. Only its local shard moves:

```python
from torch.distributed.tensor import DTensor

dt: DTensor = ...                          # A sharded parameter
handle = offload_tensor(dt, pin_memory=True)
# dt.device_mesh, dt.placements and dt.shape are unchanged here.
onload_tensor(dt, handle, device=device)
```

### Implementing `Offloadable`

A subsystem that owns GPU state implements the protocol, so that the `Trainer` can offload and onload it together with the others. A typical implementation does three things:

1.  Keep the handles in a mirror.
2.  Wait for the asynchronous copies with `torch.cuda.synchronize`.
3.  Reject a second offload or onload in a row.

```python
import torch
from d9d.core.offload import Offloadable, OffloadContext, OffloadedTensor, OnloadContext, offload_tensor, onload_tensor


class MySubsystem(Offloadable):
    def __init__(self, tensors: list[torch.Tensor]):
        self._tensors = tensors
        self._mirror: dict[int, OffloadedTensor] | None = None

    def offload(self, ctx: OffloadContext) -> None:
        if self._mirror is not None:
            raise RuntimeError("MySubsystem is already offloaded.")
        self._mirror = {id(t): offload_tensor(t, pin_memory=ctx.pin_memory) for t in self._tensors}
        # Wait for the non-blocking device-to-host copies before the storage is freed.
        torch.cuda.synchronize(ctx.dist_context.current_device)

    def onload(self, ctx: OnloadContext) -> None:
        if self._mirror is None:
            raise RuntimeError("MySubsystem is not offloaded.")
        device = ctx.dist_context.current_device
        for t in self._tensors:
            onload_tensor(t, self._mirror[id(t)], device=device)
        torch.cuda.synchronize(device)
        self._mirror = None

    def is_offloaded(self) -> bool:
        return self._mirror is not None
```

The built-in implementations follow the same pattern: `TrackedModules` (model parameters and buffers), `PipelinedOptimizer` (optimizer state) and `GradientManager` (gradient buckets and the residual loss accumulator).

## API Reference

::: d9d.core.offload
