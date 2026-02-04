from torch import Tensor
from torch.distributed.tensor import DTensor


def dist_grad_from_local(data: DTensor, local_grad: Tensor) -> DTensor:
    """
    Constructs a DTensor gradient from a local tensor using data placement info.

    Args:
        data: The original parameter DTensor (source of metadata).
        local_grad: The local tensor containing gradient data.

    Returns:
        A new DTensor wrapping the local gradient.
    """

    return DTensor.from_local(
        local_grad,
        shape=data.shape,
        stride=data.stride(),
        device_mesh=data.device_mesh,
        placements=data.placements
    )
