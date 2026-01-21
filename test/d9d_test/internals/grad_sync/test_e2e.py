import pytest
import torch
from d9d.core.dist_context import DENSE_DOMAIN
from d9d.internals.grad_sync.synchronizer import GradientSynchronizer
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard


def _make_dtensor_param(
        shape: tuple[int, ...],
        value: float,
        mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        dtype: torch.dtype,
        grad_dtype: torch.dtype,
):
    local_t = torch.full(shape, value, device="cuda", dtype=dtype)
    dt = DTensor.from_local(local_t, device_mesh=mesh, placements=placements)
    param = nn.Parameter(dt)
    param.grad_dtype = grad_dtype
    return param


@pytest.mark.distributed
@pytest.mark.parametrize(
    "tensor_specs",
    [
        [
            {"shape": (8, 1), "placements": (Shard(0),), "will_sync": False},
            {"shape": (151, 55), "placements": (Replicate(),), "will_sync": True},
            {"shape": (1, 8), "placements": (Shard(1),), "will_sync": False},
            {"shape": (8, 16), "placements": (Replicate(),), "will_sync": True},
        ],
        [
            {"shape": (10, 2), "placements": (Replicate(),), "will_sync": True},
            {"shape": (151, 55), "placements": (Replicate(),), "will_sync": True},
            {"shape": (2, 3, 4, 5), "placements": (Replicate(),), "will_sync": True},
            {"shape": (8, 16), "placements": (Replicate(),), "will_sync": True},
        ]
    ]
)
@pytest.mark.parametrize(
    ("param_dtype", "grad_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16)
    ]
)
def test_e2e(dist_ctx_dpr, tensor_specs, param_dtype, grad_dtype):
    sync_mesh = dist_ctx_dpr.mesh_for(DENSE_DOMAIN)["dp_replicate"]
    params = [
        _make_dtensor_param(
            shape=spec["shape"], value=1.0,
            grad_dtype=grad_dtype, dtype=param_dtype,
            mesh=sync_mesh, placements=spec["placements"]
        )
        for spec in tensor_specs
    ]
    sync = GradientSynchronizer(param_groups=[params], bucket_size_mb=2, require_accumulations=2)

    sync.bind()

    # perform multiple steps
    for _ in range(5):
        # STAGE 0: check grad is bound and 0
        for param, spec in zip(params, tensor_specs, strict=True):
            if spec["will_sync"]:
                assert (param.grad.to_local() == 0).all().item()
            else:
                assert param.grad is None

        rank = sync_mesh.get_local_rank()

        # STAGE 1: grad = rank
        for param in params:
            (param.to_local() * rank).sum().backward()

        # it is 1 accumulation out of 2
        with pytest.raises(ValueError):
            sync.wait()

        for param in params:
            expect_grad = torch.full(param.to_local().shape, fill_value=rank, dtype=grad_dtype, device="cuda")
            assert torch.allclose(param.grad.to_local(), expect_grad)

        # STAGE 2: grad = 2rank and sync
        for param in params:
            (param.to_local() * rank).sum().backward()

        sync.wait()

        world_size = sync_mesh.size()
        # Expected Grad = 2Sum(0..W-1) = 2 * (W - 1) * W / 2 = (W - 1) * W
        for param, spec in zip(params, tensor_specs, strict=True):
            if spec["will_sync"]:
                expect_grad = torch.full(
                    param.to_local().shape,
                    fill_value=world_size * (world_size - 1),
                    dtype=grad_dtype,
                    device="cuda"
                )
                assert torch.allclose(param.grad.to_local(), expect_grad)
            else:
                expect_grad = torch.full(param.to_local().shape, fill_value=rank * 2, dtype=grad_dtype, device="cuda")
                assert torch.allclose(param.grad.to_local(), expect_grad)

        # STAGE 3: zero grad and after that repeat
        sync.zero_grad()

    # STAGE 4: unbind and check that we have cleaned up
    sync.unbind()

    for param in params:
        assert param._post_accumulate_grad_hooks is None or len(param._post_accumulate_grad_hooks) == 0
        assert param.grad is None
