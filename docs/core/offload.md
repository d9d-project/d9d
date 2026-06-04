# State Offloading

## About

The `d9d.core.offload` package provides the low-level machinery for releasing GPU-resident training state to host (CPU) memory and restoring it later. It is the foundation of the Trainer's **sleep / wake** API, which frees the accelerator for a colocated workload - most notably a rollout engine sharing the same GPUs in colocated reinforcement learning.

The package defines two things:

1. **The `Offloadable` protocol** - the contract a subsystem implements to declare that it owns GPU memory and knows how to release and reacquire it.
2. **The tensor-swap primitives** - `offload_tensor` and `onload_tensor`, which move a single tensor's storage to and from the host while preserving the tensor (and `DTensor` wrapper) object.

The high-level user entry points - `Trainer.sleep()`, `Trainer.wake()` and `Trainer.is_sleeping()` - are documented in the [Training Loop](../loop/train.md) page. This page covers the primitives those methods are built on.

## The Round-Trip Invariant

The central guarantee of this subsystem is that **an `offload` followed by an `onload` is observationally a no-op**. Across the round trip:

* Parameter and buffer **object identity** is preserved.
* Optimizer **state-dict keys** and the tensor objects they map to are preserved.
* `DTensor` **wrapper instances**, their `device_mesh`, `placements`, global `shape`, `stride` and `dtype` are preserved.

Only the *underlying device storage* is reallocated. This is what makes offloading safe in the presence of external references - gradient hooks, optimizer state keyed by parameter, or a frozen reference model held by a task all keep pointing at the same objects after waking up.

The trick is that the swap rebinds storage in place rather than creating new tensors:

* For a plain tensor, `tensor.data` is rebound to a host (then back to a device) buffer.
* For a `DTensor`, only the **local shard's** storage (`_local_tensor.data`) is rebound. All distributed metadata lives on the wrapper and never leaves it, so any code still holding the `DTensor` sees the same object with its placements intact.

## Usage

`offload_tensor` / `onload_tensor` operate on a single tensor and return an `OffloadedTensor` handle that you hold between the two calls. The same tensor object is passed to both.

```python
import torch
from d9d.core.offload import offload_tensor, onload_tensor

device = torch.device("cuda")
param = torch.randn(4096, 4096, device=device)

# Release the GPU storage; `param` now points at a host mirror.
handle = offload_tensor(param, pin_memory=True)
assert param.device.type == "cpu"

# ... colocated workload runs on the freed GPU ...

# Restore the GPU storage in place; `param` is the same object as before.
onload_tensor(param, handle, device=device)
assert param.device.type == "cuda"
```

`DTensor` is handled transparently - pass the wrapper and only its local shard is swapped:

```python
from torch.distributed.tensor import DTensor

dt: DTensor = ...                          # a sharded parameter
handle = offload_tensor(dt, pin_memory=True)
# dt.device_mesh, dt.placements, dt.shape are unchanged here
onload_tensor(dt, handle, device=device)
```

### Implementing `Offloadable`

Subsystems that own GPU state implement the protocol so the Trainer can fan offload/onload out to them as a unit. The pattern is to record the handles in a mirror, drain the asynchronous copies with `torch.cuda.synchronize`, and guard against double offload/onload.

```python
import torch
from d9d.core.offload import Offloadable, OffloadContext, OffloadedTensor, OnloadContext, offload_tensor, onload_tensor


class MySubsystem(Offloadable):
    def __init__(self, tensors: list[torch.Tensor]):
        self._tensors = tensors
        self._mirror: dict[int, OffloadedTensor] | None = None

    def offload(self, ctx: OffloadContext) -> None:
        if self._mirror is not None:
            raise RuntimeError("already offloaded")
        self._mirror = {id(t): offload_tensor(t, pin_memory=ctx.pin_memory) for t in self._tensors}
        # Drain the non-blocking device-to-host copies before storage is freed.
        torch.cuda.synchronize(ctx.dist_context.current_device)

    def onload(self, ctx: OnloadContext) -> None:
        if self._mirror is None:
            raise RuntimeError("not offloaded")
        device = ctx.dist_context.current_device
        for t in self._tensors:
            onload_tensor(t, self._mirror[id(t)], device=device)
        torch.cuda.synchronize(device)
        self._mirror = None

    def is_offloaded(self) -> bool:
        return self._mirror is not None
```

The built-in `Offloadable` implementations - `TrackedModules` (model parameters and buffers), `PipelinedOptimizer` (optimizer state) and `GradientManager` (gradient buckets and the residual loss accumulator) - all follow this shape.

## Sleep Tags

Offloading is selected by `SleepTag`, a subsystem selector shared by `Trainer.sleep` and `Trainer.wake`:

* **`SleepTag.TENSOR_STATES`** - all GPU tensor state (model parameters and buffers, optimizer state, gradient buckets and the residual loss accumulator). Offloaded as a single unit. This is the only tag enabled by `DEFAULT_SLEEP_TAGS`.
* **`SleepTag.COMMS`** - NCCL process groups. Opt-in and **not yet implemented**; requesting it raises `NotImplementedError`.

## API Reference

::: d9d.core.offload
    options:
        heading_level: 4
