from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Partial, Placement, Replicate, Shard


def map_placement_for_grad_sync(placement: Placement) -> Placement:
    """
    Determines the gradient placement state required during synchronization/accumulation.

    When accumulating gradients, Replicated data implies the gradients are Partial
    (summed across ranks).

    Args:
        placement: The placement of the data tensor.

    Returns:
        The expected placement for the gradient tensor during sync.

    Raises:
        ValueError: If the placement is unknown.
    """

    match placement:
        case Shard():
            return placement
        case Replicate():
            return Partial("sum")
        case _:
            raise ValueError(f"Unknown placement {placement}")


def dist_grad_from_local(data: DTensor, local_grad: Tensor):
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
        device_mesh=data.device_mesh,
        placements=[map_placement_for_grad_sync(x) for x in data.placements]
    )


def mark_grad_sync_complete(param: nn.Parameter):
    """
    Updates the parameter's gradient placement to reflect completed synchronization.

    Args:
        param: The parameter to update.
    """

    param.grad = DTensor.from_local(
        param.grad.to_local(),
        device_mesh=param.grad.device_mesh,
        placements=[map_placement_for_grad_after_sync(x) for x in param.grad.placements],
        run_check=False
    )


def mark_grad_sync_awaiting(param: nn.Parameter):
    """
    Updates the parameter's gradient placement to reflect a state awaiting synchronization.

    Args:
        param: The parameter to update.
    """

    param.grad = DTensor.from_local(
        param.grad.to_local(),
        device_mesh=param.grad.device_mesh,
        placements=[map_placement_for_grad_sync(x) for x in param.grad.placements],
        run_check=False
    )


def map_placement_for_grad_after_sync(placement: Placement) -> Placement:
    """
    Maps gradient placements from their accumulation state to their synchronized state.

    Args:
        placement: The placement during accumulation.

    Returns:
        The placement after reduction.

    Raises:
        ValueError: If the placement is unexpected.
    """

    match placement:
        case Shard():
            return placement
        case Partial():
            return Replicate()
        case _:
            raise ValueError(f"Unknown placement {placement}")
