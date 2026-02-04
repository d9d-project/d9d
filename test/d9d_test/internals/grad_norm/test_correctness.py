import math

import pytest
import torch
from d9d.core.dist_context import EXPERT_DOMAIN
from d9d.internals.grad_norm import clip_grad_norm_distributed_, group_parameters_for_norm
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor


@pytest.mark.local
@pytest.mark.parametrize("norm_type", [2.0, 3.0, float("inf")])
@pytest.mark.parametrize("max_norm", [1.0, 5.0, None])
def test_local_equivalence(norm_type, max_norm):
    device = torch.device("cuda")

    # Create two sets of identical parameters
    params_torch = [nn.Parameter(torch.randn(10, 10, device=device)) for _ in range(2)]
    params_d9d = [nn.Parameter(p.clone().detach()) for p in params_torch]

    for p in params_torch:
        p.grad = torch.randn_like(p)

    for p_t, p_d in zip(params_torch, params_d9d, strict=True):
        p_d.grad = p_t.grad.clone()

    # Calculate using PyTorch
    torch_total_norm = torch.nn.utils.clip_grad_norm_(
        params_torch, max_norm=max_norm or 1e9, norm_type=norm_type
    )

    # Calculate using d9d
    groups = group_parameters_for_norm(params_d9d)

    d9d_total_norm = clip_grad_norm_distributed_(
        groups, max_norm=max_norm, norm_type=norm_type, pp_mesh=None
    )

    # Check Norm Value
    assert torch.isclose(torch_total_norm, d9d_total_norm)

    # Check Gradients if clipping happened
    if max_norm is not None:
        for p_t, p_d in zip(params_torch, params_d9d, strict=True):
            assert torch.allclose(p_t.grad, p_d.grad)


@pytest.mark.distributed
def test_pp_ep_norm_calculation(dist_ctx_pp4_dpr2):
    dist_ctx = dist_ctx_pp4_dpr2

    ep_mesh = dist_ctx.mesh_for(EXPERT_DOMAIN)
    sub_mesh = ep_mesh[["ep_replicate", "ep_shard"]]
    pp_mesh = ep_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()

    param_rep = nn.Parameter(
        distribute_tensor(
            torch.zeros(8, 8, device="cuda"),
            device_mesh=sub_mesh,
            placements=[Replicate(), Replicate()]
        )
    )

    param_sharded = nn.Parameter(
        distribute_tensor(
            torch.zeros(8, 16, device="cuda"),
            device_mesh=sub_mesh,
            placements=[Replicate(), Shard(1)]
        )
    )

    params = []

    if pp_rank == 0:
        # PP0 owns the Replicated Parameter
        # Grad = 1.0 everywhere. 8x8 = 64 elements.
        # Since it is replicated, norm is calculated locally. SqNorm = 64.
        param_rep.grad = distribute_tensor(
            torch.ones(8, 8, device="cuda"),
            device_mesh=sub_mesh,
            placements=[Replicate(), Replicate()]
        )
        params.append(param_rep)

    elif pp_rank == 1:
        # PP1 owns the Sharded Parameter
        # We manually construct the local gradient to be 1.0
        # Global shape (8, 16). EP=2.
        # Local partition is (8, 8).
        # Local SqNorm = 64.
        # Since it is sharded, global norm sums the squares over the shard group.
        # Total Global SqNorm for this param = 64 (Rank EP0) + 64 (Rank EP1) = 128.
        param_sharded.grad = DTensor.from_local(
            torch.ones(8, 8, device="cuda"),
            sub_mesh,
            [Replicate(), Shard(1)]
        )
        params.append(param_sharded)

    # --- Group and Calculate ---
    groups = group_parameters_for_norm(params)

    # Expected Logic:
    # PP0 Contribution: 64.0 (Replicated param, no reduction needed horizontally)
    # PP1 Contribution: 128.0 (Sharded param, horizontal reduction over EP sums 64+64)
    # PP Reduction: Sums PP0 + PP1 = 64 + 128 = 192.
    # Final Result: Sqrt(192)

    global_norm = clip_grad_norm_distributed_(
        groups, max_norm=None, norm_type=2.0, pp_mesh=pp_mesh
    )

    expected_sq = 64.0 + 128.0
    expected_norm = math.sqrt(expected_sq)

    assert torch.isclose(global_norm, torch.tensor(expected_norm, device="cuda"))

    # --- Verify Clipping ---
    # Apply clipping with strict max_norm=1.0
    # Expected scaling factor: 1.0 / sqrt(192)
    clip_grad_norm_distributed_(
        groups, max_norm=1.0, norm_type=2.0, pp_mesh=pp_mesh
    )

    expected_scale = 1.0 / expected_norm

    if pp_rank == 0:
        assert torch.allclose(param_rep.grad.to_local(), torch.tensor(expected_scale, device="cuda"))

    elif pp_rank == 1:
        assert torch.allclose(param_sharded.grad.to_local(), torch.tensor(expected_scale, device="cuda"))


@pytest.mark.local
def test_empty_params():
    groups = group_parameters_for_norm([])
    norm = clip_grad_norm_distributed_(groups, max_norm=1.0, norm_type=2.0, pp_mesh=None)
    assert norm.item() == 0.0
