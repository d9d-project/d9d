from typing import TypeVar

import torch.distributed as dist

T = TypeVar('T')


def gather_object(
        obj: T,
        group: dist.ProcessGroup,
        group_dst: int
) -> list[T] | None:
    """
    Gathers picklable objects from the whole process group to a specific destination rank.

    This acts as a wrapper around torch.distributed.gather_object that automatically
    initializes the output buffer list on the destination rank.

    Args:
        obj: The local object to send. Must be picklable.
        group: The process group to work on.
        group_dst: The rank within the group that will receive the objects.

    Returns:
        A list of objects from all ranks on the destination rank; None on other ranks.
    """

    if group.rank() == group_dst:
        save_list = [None for _ in range(group.size())]
    else:
        save_list = None
    dist.gather_object(
        obj,
        save_list,
        group=group,
        group_dst=group_dst
    )
    return save_list


def all_gather_object(
        obj: T,
        group: dist.ProcessGroup
) -> list[T]:
    """
    Gathers picklable objects from the whole process group to all ranks.

    This acts as a wrapper around torch.distributed.all_gather_object that automatically
    initializes the output buffer list on all ranks.

    Args:
        obj: The local object to send. Must be picklable.
        group: The process group to work on.

    Returns:
        A list of objects containing the data gathered from all ranks.
    """

    save_list = [None for _ in range(group.size())]
    dist.all_gather_object(
        save_list,
        obj,
        group=group
    )
    return save_list
