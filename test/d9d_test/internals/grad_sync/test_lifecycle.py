import pytest
import torch
from d9d.core.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.internals.grad_sync.synchronizer import GradientSynchronizer
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate


def _make_replicated_param(mesh: DeviceMesh) -> nn.Parameter:
    local = torch.zeros(4, device="cuda", dtype=torch.float32)
    param = nn.Parameter(DTensor.from_local(local, device_mesh=mesh, placements=(Replicate(),)))
    param.grad_dtype = torch.float32
    return param


def _backward(param: nn.Parameter, value: float):
    (param.to_local() * value).sum().backward()


def _assert_grad(param: nn.Parameter, value: float):
    local_grad = param.grad.to_local()
    torch.testing.assert_close(local_grad, torch.full_like(local_grad, value), rtol=0, atol=0)


@pytest.mark.distributed
def test_failed_wait_keeps_buckets_unchanged(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate"]
    rank = mesh.get_local_rank()
    world_size = mesh.size()

    first = _make_replicated_param(mesh)
    second = _make_replicated_param(mesh)
    # The bucket of ``first`` comes first in the bucket list. A wait() that marks the buckets one by one marks it
    # before it fails on the bucket of ``second``.
    sync = GradientSynchronizer(param_groups=[[first], [second]], bucket_size_mb=1)
    sync.bind()

    try:
        sync.set_required_accumulations(1)
        _backward(first, rank + 1)

        # The bucket of ``second`` is not ready.
        with pytest.raises(ValueError, match="not ready"):
            sync.wait()

        _backward(second, rank + 1)
        sync.wait()

        expected_sum = world_size * (world_size + 1) / 2
        _assert_grad(first, expected_sum)
        _assert_grad(second, expected_sum)
    finally:
        torch.cuda.synchronize()
        sync.unbind()


@pytest.mark.distributed
def test_unused_parameter_raises_on_wait(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate"]

    used = _make_replicated_param(mesh)
    unused = _make_replicated_param(mesh)
    sync = GradientSynchronizer(param_groups=[[used, unused]], bucket_size_mb=1)
    sync.bind()

    try:
        sync.set_required_accumulations(1)
        # ``unused`` gets no gradient, so its bucket never reaches the required accumulation count.
        _backward(used, 1.0)

        with pytest.raises(ValueError, match="must get one in each backward pass"):
            sync.wait()
    finally:
        torch.cuda.synchronize()
        sync.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("wait_before_extra_backward", [False, True], ids=["before_wait", "after_wait"])
def test_extra_accumulation_raises(dist_ctx_factory, wait_before_extra_backward):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate"]

    param = _make_replicated_param(mesh)
    sync = GradientSynchronizer(param_groups=[[param]], bucket_size_mb=1)
    sync.bind()

    try:
        sync.set_required_accumulations(1)
        _backward(param, 1.0)
        if wait_before_extra_backward:
            sync.wait()

        # The extra backward pass writes into the buffer that the reduction reads or has already reduced.
        with pytest.raises(ValueError, match="more times"):
            _backward(param, 1.0)
    finally:
        torch.cuda.synchronize()
        sync.unbind()


@pytest.mark.distributed
def test_gradient_set_to_none_outside_synchronizer_raises(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate"]

    param = _make_replicated_param(mesh)
    sync = GradientSynchronizer(param_groups=[[param]], bucket_size_mb=1)
    sync.bind()

    try:
        sync.set_required_accumulations(1)
        # This is what optimizer.zero_grad(set_to_none=True) does.
        param.grad = None

        # Autograd creates a new gradient tensor, which the reduction of the buffer would miss.
        with pytest.raises(ValueError, match="no longer a view"):
            _backward(param, 1.0)
    finally:
        torch.cuda.synchronize()
        sync.unbind()
