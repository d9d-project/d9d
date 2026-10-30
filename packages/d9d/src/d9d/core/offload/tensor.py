import dataclasses

import torch
from torch.distributed.tensor import DTensor


@dataclasses.dataclass(slots=True, frozen=True)
class OffloadedTensor:
    """Handle to a tensor whose local storage was moved to host memory.

    ``offload_tensor`` produces it and ``onload_tensor`` consumes it. Subsystems hold it as an opaque handle
    between an offload and the matching onload.

    Attributes:
        host: The host buffer that holds the local storage of the offloaded tensor.
    """

    host: torch.Tensor


def _local_storage_holder(tensor: torch.Tensor) -> torch.Tensor:
    """Returns the storage-bearing tensor: the DTensor's local shard, or the tensor itself."""
    # to_local() returns a new tensor object, so rebinding its .data would not reach the DTensor.
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor  # noqa: SLF001 - need the stored shard


def offload_tensor(tensor: torch.Tensor, *, pin_memory: bool) -> OffloadedTensor:
    """Moves the local storage of ``tensor`` to host memory, in place.

    The tensor object stays the same, so external references to it stay valid. For a ``DTensor``, only the
    storage of the local shard moves. The wrapper keeps its ``device_mesh``, ``placements``, global shape and
    global stride.

    Args:
        tensor: The device tensor to offload. Can be a plain tensor or a ``DTensor``.
        pin_memory: Whether to allocate the host buffer in pinned memory.

    Returns:
        A handle holding the host buffer. Pass it to ``onload_tensor`` together with the same tensor.
    """
    local = _local_storage_holder(tensor)
    host = torch.empty_like(local, device="cpu", pin_memory=pin_memory)
    host.copy_(local, non_blocking=True)
    # Rebinding .data swaps the storage but keeps the tensor object and its DTensor wrapper.
    local.data = host
    return OffloadedTensor(host=host)


def onload_tensor(tensor: torch.Tensor, offloaded: OffloadedTensor, *, device: torch.device) -> None:
    """Moves the local storage of ``tensor`` from the host buffer back to ``device``, in place.

    The tensor object, and the ``DTensor`` wrapper, stay the same instances as before the offload. Only the
    device storage is allocated again.

    Args:
        tensor: The tensor that was passed to ``offload_tensor``.
        offloaded: The handle returned by ``offload_tensor``.
        device: The device to move the local storage to.
    """
    local = _local_storage_holder(tensor)
    fresh = torch.empty_like(local, device=device)
    fresh.copy_(offloaded.host, non_blocking=True)
    local.data = fresh
