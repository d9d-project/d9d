import pytest
import torch
from d9d.core.dist_context import DENSE_DOMAIN, DeviceMeshParameters
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


def _make_param_groups(params: list[nn.Parameter], layout: str) -> list[list[nn.Parameter]]:
    match layout:
        case "one_group":
            return [params]
        case "group_per_param":
            # One bucket per parameter, so the order of the backward passes changes the order of the buckets.
            return [[param] for param in params]
        case _:
            raise ValueError(f"Unknown layout ({layout}).")


def _backward_order(params: list[nn.Parameter], rank: int, rank_dependent: bool) -> list[nn.Parameter]:
    # Odd ranks run the backward passes in reverse, like a model whose graph depends on the data.
    if rank_dependent and rank % 2 == 1:
        return params[::-1]
    return params


@pytest.mark.distributed
@pytest.mark.parametrize("rank_dependent_order", [False, True], ids=["same_order", "rank_dependent_order"])
@pytest.mark.parametrize("param_groups_layout", ["one_group", "group_per_param"])
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
        ],
    ],
)
@pytest.mark.parametrize(
    ("param_dtype", "grad_dtype"),
    [(torch.float32, torch.float32), (torch.bfloat16, torch.float32), (torch.bfloat16, torch.bfloat16)],
)
def test_e2e(dist_ctx_factory, tensor_specs, param_dtype, grad_dtype, param_groups_layout, rank_dependent_order):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    sync_mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate"]
    params = [
        _make_dtensor_param(
            shape=spec["shape"],
            value=1.0,
            grad_dtype=grad_dtype,
            dtype=param_dtype,
            mesh=sync_mesh,
            placements=spec["placements"],
        )
        for spec in tensor_specs
    ]
    sync = GradientSynchronizer(param_groups=_make_param_groups(params, param_groups_layout), bucket_size_mb=2)

    sync.bind()
    sync.set_required_accumulations(2)

    # perform multiple steps
    for _ in range(5):
        # STAGE 0: check grad is bound and 0
        for param, spec in zip(params, tensor_specs, strict=True):
            if spec["will_sync"]:
                assert (param.grad.to_local() == 0).all().item()
            else:
                assert param.grad is None

        rank = sync_mesh.get_local_rank()
        backward_order = _backward_order(params, rank, rank_dependent_order)

        # STAGE 1: grad = rank
        for param in backward_order:
            (param.to_local() * rank).sum().backward()

        # it is 1 accumulation out of 2
        with pytest.raises(ValueError, match="not ready"):
            sync.wait()

        for param in params:
            expect_grad = torch.full(param.to_local().shape, fill_value=rank, dtype=grad_dtype, device="cuda")
            assert torch.allclose(param.grad.to_local(), expect_grad)

        # STAGE 2: grad = 2rank and sync
        for param in backward_order:
            (param.to_local() * rank).sum().backward()

        sync.wait()
        sync.scale_grads(torch.tensor(0.5, device="cuda"))

        world_size = sync_mesh.size()
        # Expected Grad = 2Sum(0..W-1) * 0.5 = (W - 1) * W / 2
        for param, spec in zip(params, tensor_specs, strict=True):
            if spec["will_sync"]:
                expect_grad = torch.full(
                    param.to_local().shape,
                    fill_value=world_size * (world_size - 1) / 2,
                    dtype=grad_dtype,
                    device="cuda",
                )
                assert torch.allclose(param.grad.to_local(), expect_grad)
            else:
                expect_grad = torch.full(param.to_local().shape, fill_value=rank, dtype=grad_dtype, device="cuda")
                assert torch.allclose(param.grad.to_local(), expect_grad)

        # STAGE 3: zero grad and after that repeat
        sync.zero_grad()

    # STAGE 4: unbind and check that we have cleaned up
    sync.unbind()

    for param in params:
        assert param._post_accumulate_grad_hooks is None or len(param._post_accumulate_grad_hooks) == 0
        assert param.grad is None


@pytest.mark.local
def test_e2e_local():
    params = [nn.Parameter(torch.randn(10, 10)) for _ in range(5)]
    for p in params:
        p.grad_dtype = torch.float32

    sync = GradientSynchronizer(param_groups=[params], bucket_size_mb=1)

    sync.bind()
    sync.set_required_accumulations(2)

    # The second step checks the gradients created after zero_grad().
    for _ in range(2):
        # Iteration 1
        loss = sum(p.sum() for p in params)
        loss.backward()

        for p in params:
            assert p.grad is not None
            assert torch.is_tensor(p.grad)

        # Wait should be a no-op for local buckets
        sync.wait()

        # Iteration 2 (Accumulation)
        loss = sum(p.sum() for p in params)
        loss.backward()

        # Gradients should be accumulated (x2) and then halved
        sync.wait()
        sync.scale_grads(torch.tensor(0.5))
        for p in params:
            assert torch.equal(p.grad, torch.ones_like(p))

        # 4. Zero Grad
        sync.zero_grad()
        for p in params:
            assert p.grad is None

    # 5. Cleanup
    sync.unbind()
