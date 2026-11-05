import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DENSE_DOMAIN, DenseDim, DenseDomainConfig, DeviceMeshParameters
from d9d.parallel import parallelize_hsdp
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard

from d9d_dev.parallel.runtime_step import step_with_runtime


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


@pytest.mark.distributed
def test_parallelize_hsdp(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4)))

    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE, DenseDim.DP_CP_SHARD]

    param_size = 16

    model = SimpleLinear(param_size).cuda()

    parallelize_hsdp(model, mesh, shard_dim="dp_cp_shard")

    for param in model.parameters():
        assert isinstance(param, DTensor)
        assert param.placements == (Replicate(), Shard(0))
        assert param.device_mesh.mesh_dim_names == ("dp_replicate", "dp_cp_shard")

    rank = dist.get_rank()
    x = torch.arange(param_size, device="cuda", dtype=torch.float) + rank

    runtime = step_with_runtime(model, lambda: model(x).backward())

    shard_group_size = mesh[DenseDim.DP_CP_SHARD].size()
    world_size = dist.get_world_size()

    global_indices = torch.arange(param_size, device="cuda", dtype=torch.float)

    # The final reduction also sums the replicas, so every rank of the world contributes.
    exp_global_w = global_indices * world_size + sum(range(world_size))
    exp_global_b = torch.full((param_size,), float(world_size), device="cuda")

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
    assert model.w.param.grad.placements == (Replicate(), Shard(0))
    assert model.b.param.grad.placements == (Replicate(), Shard(0))

    assert torch.allclose(model.w.param.grad.to_local(), exp_local_w)
    assert torch.allclose(model.b.param.grad.to_local(), exp_local_b)
    runtime.unbind()


@pytest.mark.distributed
def test_parallelize_hsdp_keeps_mesh_order(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(
        DeviceMeshParameters(data_parallel=4, context_parallel=2, dense=DenseDomainConfig(data_parallel_shard=4))
    )
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE, DenseDim.DP_CP_SHARD, DenseDim.CP_REPLICATE]
    model = SimpleLinear(16).cuda()

    parallelize_hsdp(model, mesh, shard_dim=DenseDim.DP_CP_SHARD)

    for param in model.parameters():
        assert param.device_mesh.mesh_dim_names == ("dp_cp_shard", "cp_replicate")
        assert param.placements == (Shard(0), Replicate())
