import pytest
import torch
import torch.distributed as dist
from d9d.core.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.module.parallelism.api import parallelize_hsdp
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Replicate, Shard


class NestedModule(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.param = nn.Parameter(torch.ones(size, dtype=torch.float))

    def forward(self):
        return self.param


class SimpleLinear(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.w = NestedModule(size)
        self.b = NestedModule(size)

    def forward(self, x):
        return (x * self.w() + self.b()).sum()


def _sum_peer_ranks(rank: int, shard_group_size: int, shard_group: int) -> int:
    my_rank_tensor = torch.tensor([rank], device="cuda")
    group_ranks = [torch.zeros_like(my_rank_tensor) for _ in range(shard_group_size)]
    dist.all_gather(group_ranks, my_rank_tensor, group=shard_group)
    sum_peer_ranks = sum(r.item() for r in group_ranks)
    return sum_peer_ranks


@pytest.mark.distributed
def test_parallelize_hsdp(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(
        data_parallel_shard=4,
        data_parallel_replicate=2
    ))

    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate", "dp_cp_shard"]

    param_size = 16

    model = SimpleLinear(param_size).cuda()

    parallelize_hsdp(model, mesh, shard_dim="dp_cp_shard")

    assert isinstance(model, FSDPModule)
    for param in model.parameters():
        assert isinstance(param, DTensor)
        assert param.placements == (Shard(0), Replicate())
        assert param.device_mesh.mesh_dim_names == ("dp_cp_shard", "dp_replicate")

    rank = dist.get_rank()
    x = torch.arange(param_size, device="cuda", dtype=torch.float) + rank

    y = model(x)
    y.backward()

    shard_dim_idx = mesh.mesh_dim_names.index("dp_cp_shard")
    shard_group = mesh.get_group(shard_dim_idx)
    shard_group_size = mesh.size(shard_dim_idx)

    sum_peer_ranks = _sum_peer_ranks(rank, shard_group_size, shard_group)

    global_indices = torch.arange(param_size, device="cuda", dtype=torch.float)

    # Expected Global W (16 elements)
    exp_global_w = global_indices * shard_group_size + sum_peer_ranks
    # Expected Global B (16 elements)
    exp_global_b = torch.full((param_size,), float(shard_group_size), device="cuda")

    my_shard_rank = mesh.get_local_rank("dp_cp_shard")

    total_params = param_size
    params_per_shard = total_params // shard_group_size

    start_idx = my_shard_rank * params_per_shard
    end_idx = start_idx + params_per_shard

    exp_local_w = exp_global_w[start_idx:end_idx]
    exp_local_b = exp_global_b[start_idx:end_idx]

    # Check grads
    assert model.w.param.grad is not None
    assert model.b.param.grad is not None
    assert isinstance(model.w.param.grad, DTensor)
    assert isinstance(model.b.param.grad, DTensor)
    assert model.w.param.grad.placements == (Shard(0), Replicate())
    assert model.b.param.grad.placements == (Shard(0), Replicate())

    assert torch.allclose(model.w.param.grad.to_local(), exp_local_w)
    assert torch.allclose(model.b.param.grad.to_local(), exp_local_b)
