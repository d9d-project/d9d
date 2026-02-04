import math

import torch.distributed as dist
import torch.nn.utils
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor

from d9d.internals.grad_norm.group import ParametersForNorm


def _reduce_op_from_norm_type(norm_type: float) -> dist.ReduceOp.RedOpType:
    if math.isinf(norm_type):
        return dist.ReduceOp.MAX
    else:
        return dist.ReduceOp.SUM


def _parameter_to_local_grad(parameter: nn.Parameter) -> torch.Tensor:
    grad = parameter.grad

    if grad is None:
        raise ValueError("None grad detected")

    if isinstance(grad, DTensor):
        return grad.to_local()
    else:
        return grad


def _get_local_norm_pow(
        parameters: list[nn.Parameter],
        norm_type: float
) -> torch.Tensor:
    # calculates for local

    if len(parameters) == 0:
        return torch.tensor(0.0, device="cuda")

    norm_val = torch.nn.utils.get_total_norm(
        [_parameter_to_local_grad(x) for x in parameters],
        norm_type=norm_type,
        foreach=True,
        error_if_nonfinite=False
    )

    if math.isinf(norm_type):
        return norm_val
    else:
        return norm_val ** norm_type


def _get_global_norm_pow_horizontal(
        parameter_groups: ParametersForNorm,
        norm_type: float
) -> torch.Tensor:
    # calculates for horizontal parallelism
    if len(parameter_groups) == 0:
        return torch.tensor(0.0, device="cuda")

    norms: list[torch.Tensor] = []
    works: list[dist.Work] = []
    for group, group_params in parameter_groups.items():
        local_norm_pow = _get_local_norm_pow(group_params, norm_type=norm_type)
        if group.shard_meshes is not None:
            if len(group.shard_meshes) != 1:
                raise ValueError(
                    "Currently we do not support calculating norm for tensors that are sharded on multiple dims - feel "
                    "free to file an issue if you need it."
                )
            process_group = group.shard_meshes[0].get_group()
            work = dist.all_reduce(
                local_norm_pow,
                op=_reduce_op_from_norm_type(norm_type),
                group=process_group,
                async_op=True
            )
            works.append(work)
        norms.append(local_norm_pow)

    for work in works:
        work.wait()

    norms_total = torch.stack(norms, dim=0)

    if math.isinf(norm_type):
        return norms_total.max()
    else:
        return norms_total.sum()


def _get_global_norm_pow_pp(
        parameter_groups: ParametersForNorm,
        norm_type: float,
        pp_mesh: DeviceMesh | None
) -> torch.Tensor:
    norm = _get_global_norm_pow_horizontal(
        parameter_groups=parameter_groups,
        norm_type=norm_type
    )
    if pp_mesh is not None:
        dist.all_reduce(norm, op=_reduce_op_from_norm_type(norm_type), group=pp_mesh.get_group())
    return norm


def _clip_grad_with_norm_(
        parameter_groups: ParametersForNorm,
        max_norm: float,
        total_norm: torch.Tensor
):
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for group in parameter_groups.values():
        grads = [_parameter_to_local_grad(x) for x in group]
        torch._foreach_mul_(grads, clip_coef_clamped)


def clip_grad_norm_distributed_(
        parameter_groups: ParametersForNorm,
        max_norm: float | None,
        norm_type: float,
        pp_mesh: DeviceMesh | None
) -> torch.Tensor:
    """
    Clips gradient norms in a fully distributed environment.

    This function calculates the global gradient norm across all dimensions of parallelism
    (Horizontal - DP/CP/TP/EP/..., and Pipeline) and scales the gradients in-place to ensure the norm
    does not exceed max_norm.

    It accurately handles DTensors by identifying their sharding placements and performing
    reductions only on the necessary process groups.

    Overlaps communication and computation if possible.

    Args:
        parameter_groups: Dictionary grouping parameters by synchronization requirements,
            typically created by `group_parameters_for_norm`.
        max_norm: The maximum allowed norm of the gradients. If None, the function
            calculates and returns the global norm without modifying the gradients.
        norm_type: The type of the norm to calculate (e.g., 2.0 for L2 norm, inf for max norm).
        pp_mesh: The device mesh representing the pipeline parallel dimension, needed
            to reduce norms across pipeline stages.

    Returns:
        The calculated global gradient norm.
    """

    with record_function("Gradient Clipping"):
        global_norm_pow = _get_global_norm_pow_pp(
            parameter_groups=parameter_groups,
            norm_type=norm_type,
            pp_mesh=pp_mesh
        )
        if math.isinf(norm_type):
            global_norm = global_norm_pow
        else:
            global_norm = global_norm_pow ** (1.0 / norm_type)

        if max_norm:
            _clip_grad_with_norm_(
                parameter_groups,
                max_norm=max_norm,
                total_norm=global_norm
            )

        return global_norm
