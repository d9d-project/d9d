import pytest
import torch
import torch.distributed as dist
from d9d.core.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.module.parallelism.api import parallelize_fsdp
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Shard


class NestedModule(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.param = nn.Parameter(torch.arange(size, dtype=torch.float))

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
@pytest.mark.parametrize(
    "mesh_dim",
    [
        "dp_cp_shard",
    ],
)
def test_parallelize_fsdp(dist_ctx_factory, mesh_dim):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_shard=4, context_parallel_shard=2))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)[mesh_dim]

    # Ensure divisibility. Mesh size 8 -> Params 16 -> 2 per Rank.
    param_size = mesh.size() * 2

    model = SimpleLinear(param_size).cuda()

    parallelize_fsdp(model, mesh)

    # Check structure
    assert isinstance(model, FSDPModule)
    for param in model.parameters():
        assert isinstance(param, DTensor)
        assert param.placements == (Shard(0),)
        assert param.device_mesh == mesh

    # Run
    rank = dist.get_rank()
    # Create input x that varies by index i AND by rank.
    # x_i = i + rank
    # This ensures grad_w varies per element.
    x = torch.arange(param_size, device="cuda", dtype=torch.float) + rank

    y = model(x)
    y.backward()

    # Calculate expected grads
    world_size = mesh.size()

    # Global Grad Calculation:
    # dL/dW_i = x_i = i + rank
    # Global Grad_i = sum_{rank=0..7}(i + rank)
    #               = world_size * i + sum_{rank=0..7}(rank)

    sum_ranks = sum(range(world_size))  # constant

    global_indices = torch.arange(param_size, device="cuda", dtype=torch.float)
    expected_global_grad_w = global_indices * world_size + sum_ranks

    expected_global_grad_b = torch.full((param_size,), float(world_size), device="cuda")

    # Slice expectations for local shard
    # Each rank holds [rank*local_size : (rank+1)*local_size]
    local_size = param_size // world_size
    start_idx = rank * local_size
    end_idx = start_idx + local_size

    expected_grad_w_local = expected_global_grad_w[start_idx:end_idx]
    expected_grad_b_local = expected_global_grad_b[start_idx:end_idx]

    # Check grads
    assert model.w.param.grad is not None
    assert model.b.param.grad is not None
    assert isinstance(model.w.param.grad, DTensor)
    assert isinstance(model.b.param.grad, DTensor)
    assert model.w.param.grad.placements == (Shard(0),)
    assert model.b.param.grad.placements == (Shard(0),)

    assert torch.allclose(model.w.param.grad.to_local(), expected_grad_w_local)

    assert torch.allclose(model.b.param.grad.to_local(), expected_grad_b_local)
