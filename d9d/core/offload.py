import dataclasses
from enum import StrEnum
from typing import Protocol, runtime_checkable

import torch
from torch.distributed.tensor import DTensor

from d9d.core.dist_context import DistributedContext


class SleepTag(StrEnum):
    """
    Subsystem selector for Trainer.sleep and Trainer.wake.

    Attributes:
        TENSOR_STATES: All GPU tensor state - model parameters and buffers, optimizer state,
            gradient buckets and the residual loss accumulator. Always offloaded as a unit.
        COMMS: NCCL process groups. Opt-in; its implementation is deferred to a second phase,
            so requesting it currently raises NotImplementedError.
    """

    TENSOR_STATES = "tensor_states"
    COMMS = "comms"


DEFAULT_SLEEP_TAGS = frozenset({SleepTag.TENSOR_STATES})
"""The default tag set for Trainer.sleep and Trainer.wake: tensor state only, no comms."""


@dataclasses.dataclass(kw_only=True, frozen=True)
class OffloadContext:
    """
    Context passed to Offloadable.offload.

    Attributes:
        dist_context: The distributed context the subsystem was built under.
        host_device: Destination host device (always torch.device("cpu")).
        pin_memory: Whether to allocate the host buffer in pinned memory.
    """

    dist_context: DistributedContext
    host_device: torch.device
    pin_memory: bool


@dataclasses.dataclass(kw_only=True, frozen=True)
class OnloadContext:
    """
    Context passed to Offloadable.onload.

    Attributes:
        dist_context: The distributed context the subsystem was built under.
        device: Target accelerator device for the restored tensors.
    """

    dist_context: DistributedContext
    device: torch.device


@runtime_checkable
class Offloadable(Protocol):
    """
    Protocol for subsystems that own GPU-resident state and can release it to host memory.

    An "offload" followed by an "onload" must be observationally a no-op: parameter identities,
    optimizer state keys, DTensor wrapper instances, placements and dtypes are all preserved
    across the round trip. Only the underlying device storages are reallocated.
    """

    def offload(self, ctx: OffloadContext) -> None:
        """
        Releases the GPU memory owned by this subsystem, moving its state to host memory.

        Args:
            ctx: Context for this operation.
        """

    def onload(self, ctx: OnloadContext) -> None:
        """
        Restores GPU residency of the state previously released by "offload".

        Args:
            ctx: Context for this operation.
        """

    def is_offloaded(self) -> bool:
        """
        Reports whether this subsystem currently has its state on host memory.

        Returns:
            True if the subsystem is offloaded, False otherwise.
        """


@dataclasses.dataclass(slots=True, frozen=True)
class OffloadedTensor:
    """
    Handle to a tensor whose local storage has been swapped to host memory.

    Produced by "offload_tensor" and consumed by "onload_tensor". Subsystems hold these as
    opaque handles between an offload and the matching onload.

    Attributes:
        host: The host-memory mirror that currently backs the offloaded tensor's local storage.
    """

    host: torch.Tensor


def _local_storage_holder(tensor: torch.Tensor) -> torch.Tensor:
    """Returns the storage-bearing tensor: the DTensor's local shard, or the tensor itself."""
    if isinstance(tensor, DTensor):
        return tensor._local_tensor  # noqa: SLF001
    return tensor


def offload_tensor(tensor: torch.Tensor, *, pin_memory: bool) -> OffloadedTensor:
    """
    Swaps "tensor"'s local storage for a host-memory mirror, in place.

    For a DTensor, the wrapper instance is preserved across the swap - only the local shard's
    underlying storage is rebound, via "_local_tensor.data". "device_mesh", "placements",
    global "shape" and global "stride" live on the wrapper and never leave it, so any
    external reference to the DTensor keeps pointing at the same object. For a plain tensor,
    "tensor.data" itself is rebound. PyTorch exposes no public setter for a DTensor's local
    storage; rebuilding via "DTensor.from_local(...)" would create a new wrapper and silently
    strand external references, which is the path this design rules out.

    Args:
        tensor: The device tensor to offload. May be a plain tensor or a DTensor.
        pin_memory: Whether to allocate the host buffer in pinned memory.

    Returns:
        A handle holding the host buffer; pass back to "onload_tensor" with the same tensor.
    """

    local = _local_storage_holder(tensor)
    host = torch.empty_like(local, device="cpu", pin_memory=pin_memory)
    host.copy_(local, non_blocking=True)
    local.data = host
    return OffloadedTensor(host=host)


def onload_tensor(tensor: torch.Tensor, offloaded: OffloadedTensor, *, device: torch.device) -> None:
    """
    Restores "tensor"'s local storage to "device", in place, from a host buffer.

    The tensor object (and, for a DTensor, its wrapper) is the same instance as before the
    offload; only the underlying device storage is freshly allocated.

    Args:
        tensor: The same tensor previously passed to "offload_tensor".
        offloaded: The handle returned by "offload_tensor".
        device: The accelerator device to restore the local storage onto.
    """

    local = _local_storage_holder(tensor)
    fresh = torch.empty_like(local, device=device)
    fresh.copy_(offloaded.host, non_blocking=True)
    local.data = fresh
