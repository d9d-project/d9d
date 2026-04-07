from collections.abc import Sequence

import torch
import torch.distributed as dist
from d9d.core.dist_context import BATCH_DOMAIN, DENSE_DOMAIN, DistributedContext
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from .compare import assert_angle_and_norm_close
from .tolerances import grad_tolerance_for

_DEFAULT_DOMAIN = DENSE_DOMAIN
_DEFAULT_DENSE_REDUCE_DIMS: tuple[str, ...] = (
    "dp_replicate",
    "dp_cp_shard",
    "cp_replicate",
    "tp",
)


def shard_batch_dim(tensor: torch.Tensor, dist_ctx: DistributedContext) -> torch.Tensor:
    return (
        DTensor.from_local(
            tensor,
            device_mesh=dist_ctx.mesh_for(BATCH_DOMAIN)["dp"],
            placements=[Replicate()],
        )
        .redistribute(placements=(Shard(0),))
        .to_local()
    )


def all_reduce_over_mesh_groups(
    tensor: torch.Tensor,
    dist_ctx: DistributedContext,
    domain: str = DENSE_DOMAIN,
    dims: Sequence[str] = _DEFAULT_DENSE_REDUCE_DIMS,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> None:
    mesh = dist_ctx.mesh_for(domain)[tuple(dims)]
    for group in mesh.get_all_groups():
        dist.all_reduce(tensor, op=op, group=group)


def microbatch_slice(
    x: torch.Tensor,
    microbatch_idx: int,
    n_microbatches: int,
) -> torch.Tensor:
    if x.ndim == 0:
        return x

    assert x.shape[0] % n_microbatches == 0, (
        f"Local batch size ({x.shape[0]}) must be divisible by n_microbatches ({n_microbatches})."
    )

    microbatch_size = x.shape[0] // n_microbatches
    start = microbatch_idx * microbatch_size
    end = start + microbatch_size
    return x[start:end]


@torch.no_grad()
def sync_grads_manually(module: nn.Module):
    for param in module.parameters():
        grad = param.grad
        assert isinstance(grad, DTensor)
        sync_groups = []
        for i, placement in enumerate(grad.placements):
            if isinstance(placement, Replicate):
                sync_groups.append(grad.device_mesh.get_group(i))
            elif isinstance(placement, Shard):
                pass
            else:
                raise ValueError("Unknown placement")
        for group in sync_groups:
            dist.all_reduce(grad.to_local(), op=dist.ReduceOp.SUM, group=group)


@torch.no_grad()
def copy_params_local_to_dist(local: torch.nn.Module, distributed: torch.nn.Module):
    local_params = dict(local.named_parameters())

    for name, dist_param in distributed.named_parameters():
        assert name in local_params

        local_param = local_params[name]

        local_data = local_param.data

        assert isinstance(dist_param, DTensor)

        sharded_local_data = distribute_tensor(
            local_data, device_mesh=dist_param.device_mesh, placements=dist_param.placements
        )

        dist_param.to_local().copy_(sharded_local_data.to_local())


def check_grad_distance_all_local_dist(local_module: torch.nn.Module, dist_module: torch.nn.Module):
    local_params = dict(local_module.named_parameters())

    for name, dist_param in dist_module.named_parameters():
        assert name in local_params

        local_param = local_params[name]

        assert (local_param.grad is None) == (dist_param.grad is None)

        if local_param.grad is None:
            continue

        local_grad = local_param.grad
        assert isinstance(dist_param.grad, DTensor)
        dist_grad = dist_param.grad.full_tensor()

        assert_angle_and_norm_close(local_grad, dist_grad, tol=grad_tolerance_for(local_grad.dtype), name=name)
