import torch

from d9d.core import pytree
from d9d.core.types import MicrobatchPack


def move_pack_to_device(pack: MicrobatchPack, device: torch.types.Device) -> MicrobatchPack:
    """Moves every tensor in a microbatch pack to the target device.

    Args:
        pack: The pack of microbatches yielded by the stream (CPU, optionally pinned).
        device: The target device.

    Returns:
        A new pack with all tensors moved to the device.
    """
    return [pytree.tree_map_only(torch.Tensor, lambda x: x.to(device), microbatch) for microbatch in pack]
