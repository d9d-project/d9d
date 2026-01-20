from collections.abc import Sequence

import torch
import torch.distributed as dist


def gather(
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
        group_dst: int,
        async_op: bool = False
) -> list[torch.Tensor] | tuple[list[torch.Tensor] | None, dist.Work]:
    """
    Gathers tensors from the process group to a specific destination rank.

    This function assumes that tensors on all ranks have the same shape and dtype
    as the tensor on the current rank. It automatically allocates the output
    buffer list on the destination.

    Args:
        tensor: The local tensor to send.
        group: The process group to work on.
        group_dst: The rank within the group that will receive the tensors.
        async_op: Whether the operation should be asynchronous.

    Returns:
        If async_op is False: A list of tensors on the destination rank, None elsewhere.
        If async_op is True: A tuple containing (buffer_list, work_handle).
    """

    if group.rank() == group_dst:
        save_list = [torch.empty_like(tensor) for _ in range(group.size())]
    else:
        save_list = None

    work = dist.gather(
        tensor,
        save_list,
        group=group,
        group_dst=group_dst,
        async_op=async_op
    )

    if async_op:
        return save_list, work
    else:
        return save_list


def all_gather(
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False
) -> list[torch.Tensor] | tuple[list[torch.Tensor], dist.Work]:
    """
    Gathers tensors from the whole process group to all ranks.

    This function assumes that tensors on all ranks have the same shape and dtype
    as the tensor on the current rank. It automatically allocates the output
    buffer list.

    Args:
        tensor: The local tensor to send.
        group: The process group to work on.
        async_op: Whether the operation should be asynchronous.

    Returns:
        If async_op is False: A list of gathered tensors.
        If async_op is True: A tuple containing (buffer_list, work_handle).
    """

    save_list = [torch.empty_like(tensor) for _ in range(group.size())]
    work = dist.all_gather(
        save_list,
        tensor,
        group=group,
        async_op=async_op
    )
    if async_op:
        return save_list, work
    else:
        return save_list


def _all_gather_shapes(
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
) -> Sequence[torch.Tensor]:
    all_ndim = [torch.empty((), dtype=torch.long, device=tensor.device) for _ in range(group.size())]
    all_ndim_wait = dist.all_gather(
        all_ndim,
        torch.tensor(tensor.ndim, dtype=torch.long, device=tensor.device),
        group=group,
        async_op=True
    )
    all_ndim_wait.wait()

    all_shape = [torch.empty(ndim, dtype=torch.long, device=tensor.device) for ndim in all_ndim]
    all_shape_wait = dist.all_gather(
        all_shape,
        torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device),
        group=group,
        async_op=True
    )
    all_shape_wait.wait()

    return all_shape


def all_gather_variadic_shape(
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False
) -> list[torch.Tensor] | tuple[list[torch.Tensor], dist.Work]:
    """
    Gathers tensors of different shapes from the whole process group to all ranks.

    Unlike standard all_gather, this function first communicates the shape of the
    tensor on every rank allowing for dynamic sizing.

    Args:
        tensor: The local tensor to send.
        group: The process group to work on.
        async_op: Whether the final data gathering should be asynchronous.
                  Note that shape gathering is always synchronous.

    Returns:
        If async_op is False: A list of gathered tensors of varying shapes.
        If async_op is True: A tuple containing (buffer_list, work_handle).
    """

    all_shape = _all_gather_shapes(tensor, group)

    all_result = [torch.empty(tuple(shape), dtype=tensor.dtype, device=tensor.device) for shape in all_shape]
    all_result_wait = dist.all_gather(
        all_result,
        tensor,
        group=group,
        async_op=async_op
    )
    if async_op:
        return all_result, all_result_wait
    else:
        return all_result


def gather_variadic_shape(
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
        group_dst: int
) -> list[torch.Tensor] | None:
    """
    Gathers tensors of different shapes from the process group to a specific rank.

    This function coordinates shape exchange and uses point-to-point communication
    (isend/irecv) to gather tensors that may differ in shape across ranks.

    Currently, does not support async_op.

    Args:
        tensor: The local tensor to send.
        group: The process group to work on.
        group_dst: The rank within the group that will receive the tensors.

    Returns:
        A list of tensors of varying shapes on the destination rank; None on other ranks.
    """

    is_current_dst = group.rank() == group_dst

    all_shape = _all_gather_shapes(tensor, group)

    if is_current_dst:
        all_recv_futures = []
        all_result = [None for _ in range(group.size())]
        for group_src_i in range(group.size()):
            if group_src_i == group_dst:
                all_result[group_src_i] = tensor
                continue
            all_result[group_src_i] = torch.empty(
                tuple(all_shape[group_src_i]), dtype=tensor.dtype, device=tensor.device
            )
            all_recv_futures.append(dist.irecv(all_result[group_src_i], group=group, group_src=group_src_i))
        for recv_future in all_recv_futures:
            recv_future.wait()
        return all_result
    else:
        dist.isend(tensor=tensor, group=group, group_dst=0)
        return None
