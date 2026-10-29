import pytest
import torch
from d9d.core.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.internals.grad_sync.synchronizer import GradientSynchronizer
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

# Long enough for the reduction to start before the delayed gradient write runs.
_DELAY_CYCLES = 200_000_000


def _make_replicated_param(mesh: DeviceMesh) -> nn.Parameter:
    local = torch.zeros(4, device="cuda", dtype=torch.float32)
    param = nn.Parameter(DTensor.from_local(local, device_mesh=mesh, placements=(Replicate(),)))
    param.grad_dtype = torch.float32
    return param


@torch.no_grad()
def _accumulate_like_fsdp(param: nn.Parameter, value: float, stream: torch.cuda.Stream, delay: bool):
    # FSDP enqueues the gradient write on its own reduce-scatter stream and calls the post-accumulate hooks
    # right away, before the write runs on the GPU.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if delay:
            torch.cuda._sleep(_DELAY_CYCLES)
        param.grad.to_local().add_(value)
        for hook in param._post_accumulate_grad_hooks.values():
            hook(param)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "param_groups_layout",
    [
        # Both parameters share a bucket, and the delayed write is not the last one.
        "one_bucket",
        # The delayed bucket is ready first but starts from the hook of the bucket before it.
        "two_buckets",
    ],
)
def test_reduction_waits_for_writes_on_all_streams(dist_ctx_factory, param_groups_layout):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate"]
    rank = mesh.get_local_rank()
    world_size = mesh.size()

    first = _make_replicated_param(mesh)
    second = _make_replicated_param(mesh)
    match param_groups_layout:
        case "one_bucket":
            param_groups = [[first, second]]
            # The slow write comes first, the fast write makes the bucket ready.
            writes = [(first, True), (second, False)]
        case "two_buckets":
            # Buckets reduce in reverse group order, so the bucket of ``first`` is the first to reduce.
            param_groups = [[second], [first]]
            # The bucket of ``second`` is ready first, but it must wait for the bucket of ``first`` to start.
            writes = [(second, True), (first, False)]
        case _:
            raise ValueError(f"Unknown layout ({param_groups_layout}).")

    sync = GradientSynchronizer(param_groups=param_groups, bucket_size_mb=1)
    sync.bind()
    slow_stream = torch.cuda.Stream()
    fast_stream = torch.cuda.Stream()

    try:
        sync.set_required_accumulations(1)
        # The second step checks that zero_grad() resets the recorded writes.
        for _ in range(2):
            for param, delay in writes:
                _accumulate_like_fsdp(param, rank + 1, slow_stream if delay else fast_stream, delay)

            sync.wait()
            # The step is over, so a full sync here cannot hide a missing dependency.
            torch.cuda.synchronize()

            expected_sum = world_size * (world_size + 1) / 2
            for param in (first, second):
                local_grad = param.grad.to_local()
                torch.testing.assert_close(local_grad, torch.full_like(local_grad, expected_sum), rtol=0, atol=0)

            sync.zero_grad()
    finally:
        sync.unbind()
