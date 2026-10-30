from torch import Tensor
from torch.distributed.tensor import DTensor


def dist_grad_from_local(data: DTensor, local_grad: Tensor) -> DTensor:
    """Builds a ``DTensor`` gradient from a local tensor, with the shape, stride and placements of ``data``.

    Args:
        data: The parameter ``DTensor`` to take the metadata from.
        local_grad: The local tensor containing gradient data.

    Returns:
        A new DTensor wrapping the local gradient.
    """
    return DTensor.from_local(
        local_grad, shape=data.shape, stride=data.stride(), device_mesh=data.device_mesh, placements=data.placements
    )
