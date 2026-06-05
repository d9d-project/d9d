import dataclasses

import torch
from torch.distributed.tensor import DTensor


@dataclasses.dataclass(slots=True, frozen=True)
class OffloadedTensor:
    """Handle to a tensor whose local storage has been swapped to host memory.

    Produced by "offload_tensor" and consumed by "onload_tensor". Subsystems hold these as
    opaque handles between an offload and the matching onload.

    Attributes:
        host: The host-memory mirror that currently backs the offloaded tensor's local storage.
    """

    host: torch.Tensor


def _local_storage_holder(tensor: torch.Tensor) -> torch.Tensor:
    """Returns the storage-bearing tensor: the DTensor's local shard, or the tensor itself."""
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor  # noqa: SLF001


def offload_tensor(tensor: torch.Tensor, *, pin_memory: bool) -> OffloadedTensor:
    """Swaps "tensor"'s local storage for a host-memory mirror, in place.

    For a DTensor, the wrapper instance is preserved across the swap - only the local shard's
    underlying storage is rebound, via "_local_tensor.data". "device_mesh", "placements",
    global "shape" and global "stride" live on the wrapper and never leave it, so any
    external reference to the DTensor keeps pointing at the same object. For a plain tensor,
    "tensor.data" itself is rebound.

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
    """Restores "tensor"'s local storage to "device", in place, from a host buffer.

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
