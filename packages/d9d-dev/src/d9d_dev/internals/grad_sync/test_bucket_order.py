import pytest
import torch
import torch.distributed as dist
from d9d.core.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.internals.grad_sync.synchronizer import GradientSynchronizer
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

_NUM_BUCKETS = 3
# Per-bucket gradient scales keep every possible mix of buckets distinguishable and exact in float32.
_GRAD_SCALES = (1.0, 100.0, 10000.0)


def _make_replicated_param(mesh: DeviceMesh) -> nn.Parameter:
    local = torch.zeros(4, device="cuda", dtype=torch.float32)
    param = nn.Parameter(DTensor.from_local(local, device_mesh=mesh, placements=(Replicate(),)))
    param.grad_dtype = torch.float32
    return param


def _backward_order(rank: int, step: int, rank_dependent: bool) -> list[int]:
    order = list(range(_NUM_BUCKETS))
    if not rank_dependent:
        return order
    shift = (rank + step) % _NUM_BUCKETS
    order = order[shift:] + order[:shift]
    if rank % 2 == 1:
        order.reverse()
    return order


def _expected_events(order: list[int], accumulations: int) -> list[tuple[str, int]]:
    # A bucket starts its reduction once it and all buckets before it are ready.
    counts = [0] * _NUM_BUCKETS
    next_reduction = 0
    events = []
    for _ in range(accumulations):
        for bucket_i in order:
            events.append(("backward", bucket_i))
            counts[bucket_i] += 1
            while next_reduction < _NUM_BUCKETS and counts[next_reduction] == accumulations:
                events.append(("all_reduce", next_reduction))
                next_reduction += 1
    return events


@pytest.mark.distributed
@pytest.mark.parametrize("accumulations", [1, 2])
@pytest.mark.parametrize("rank_dependent_order", [False, True], ids=["same_order", "rank_dependent_order"])
def test_buckets_reduce_in_bucket_order(dist_ctx_factory, monkeypatch, accumulations, rank_dependent_order):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate"]
    rank = mesh.get_local_rank()
    world_size = mesh.size()

    params = [_make_replicated_param(mesh) for _ in range(_NUM_BUCKETS)]
    # One parameter group per parameter guarantees one bucket per parameter. Buckets reduce in reverse group
    # order, so reversed groups make the buckets reduce in the order of ``params``.
    sync = GradientSynchronizer(param_groups=[[param] for param in params[::-1]], bucket_size_mb=1)
    sync.bind()

    events: list[tuple[str, int]] = []
    # Each bucket holds one parameter, so its buffer starts at the gradient of that parameter.
    buffer_to_bucket = {param.grad.to_local().data_ptr(): i for i, param in enumerate(params)}
    all_reduce = dist.all_reduce

    def _recording_all_reduce(tensor, *args, **kwargs):
        events.append(("all_reduce", buffer_to_bucket[tensor.data_ptr()]))
        return all_reduce(tensor, *args, **kwargs)

    monkeypatch.setattr(dist, "all_reduce", _recording_all_reduce)

    try:
        sync.set_required_accumulations(accumulations)
        # The second step checks that zero_grad() restarts the bucket order.
        for step in range(2):
            order = _backward_order(rank, step, rank_dependent_order)

            for _ in range(accumulations):
                for bucket_i in order:
                    events.append(("backward", bucket_i))
                    (params[bucket_i].to_local() * (rank + 1) * _GRAD_SCALES[bucket_i]).sum().backward()

            sync.wait()

            rank_sum = world_size * (world_size + 1) / 2
            for bucket_i, param in enumerate(params):
                local_grad = param.grad.to_local()
                expected = torch.full_like(local_grad, rank_sum * accumulations * _GRAD_SCALES[bucket_i])
                torch.testing.assert_close(local_grad, expected, rtol=0, atol=0)

            assert events == _expected_events(order, accumulations)

            sync.zero_grad()
            events.clear()
    finally:
        sync.unbind()
