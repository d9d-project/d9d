import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.parallel import parallelize_replicate
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from d9d_dev.parallel.runtime_step import step_with_runtime


class NestedModule(nn.Module):
    def __init__(self, init_p: float):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(init_p))

    def forward(self):
        return self.param


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = NestedModule(2.0)
        self.b = NestedModule(1.0)

    def forward(self, x):
        return x * self.w() + self.b()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh_dims", [("dp_replicate",), ("dp_replicate", "cp_replicate")])
def test_parallelize_replicate(dist_ctx_factory, mesh_dims):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=4, context_parallel=2))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)[tuple(mesh_dims)]

    model = SimpleLinear().cuda()
    parallelize_replicate(model, mesh)

    # Check Structure
    for param in model.parameters():
        assert isinstance(param, DTensor)
        assert len(param.placements) == len(mesh_dims)
        for p in param.placements:
            assert isinstance(p, Replicate)

    # Run
    rank = dist.get_rank()
    x = torch.tensor(float(rank)).cuda()
    runtime = step_with_runtime(model, lambda: model(x).backward())

    # The gradients are summed over the ranks of the mesh.
    replicas = torch.tensor([float(rank), 1.0], device="cuda")
    for dim in range(mesh.ndim):
        dist.all_reduce(replicas, group=mesh.get_group(dim))
    expected_grad_w_local = replicas[0]
    expected_grad_b_local = replicas[1]

    # Check grad structure
    assert model.w.param.grad is not None
    assert model.b.param.grad is not None
    assert isinstance(model.w.param.grad, DTensor)
    assert isinstance(model.b.param.grad, DTensor)
    for p in model.w.param.grad.placements:
        assert isinstance(p, Replicate)
    for p in model.b.param.grad.placements:
        assert isinstance(p, Replicate)

    assert torch.allclose(model.w.param.grad.to_local(), expected_grad_w_local)
    assert torch.allclose(model.b.param.grad.to_local(), expected_grad_b_local)
    runtime.unbind()
