import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, distribute_tensor


def check_grad(my_grad: Tensor, hf_grad: torch.Tensor, atol: float, rtol: float):
    assert torch.allclose(my_grad, hf_grad, atol=atol, rtol=rtol)


def check_grad_distance(
        my_grad: torch.Tensor,
        hf_grad: torch.Tensor,
        tol_angle: float = 0.05,
        tol_norm_abs: float = 3e-3,
        tol_norm_rel: float = 0.1
):
    my_grad_fp32 = my_grad.float()
    hf_grad_fp32 = hf_grad.float()

    grad_non_zero_filter = ~(
            (my_grad_fp32.abs().sum(dim=-1) == 0) & (hf_grad_fp32.abs().sum(dim=-1) == 0)
    )
    my_grad_fp32 = my_grad_fp32[grad_non_zero_filter]
    hf_grad_fp32 = hf_grad_fp32[grad_non_zero_filter]

    angle_max_error_percentage = (1 - torch.cosine_similarity(my_grad_fp32, hf_grad_fp32, dim=-1)).max().item()

    hf_grad_norm = hf_grad_fp32.norm(dim=-1, p=2)
    my_grad_norm = my_grad_fp32.norm(dim=-1, p=2)

    assert angle_max_error_percentage <= tol_angle
    assert torch.allclose(my_grad_norm, hf_grad_norm, atol=tol_norm_abs, rtol=tol_norm_rel)


def check_grad_distance_all_local_dist(
        local: torch.nn.Module,
        dist: torch.nn.Module,
        tol_angle: float = 0.05,
        tol_norm_abs: float = 3e-3,
        tol_norm_rel: float = 0.1
):
    local_params = dict(local.named_parameters())

    for name, dist_param in dist.named_parameters():
        assert name in local_params

        local_param = local_params[name]

        assert (local_param.grad is None) == (dist_param.grad is None)

        if local_param.grad is None:
            continue

        local_grad = local_param.grad
        assert isinstance(dist_param.grad, DTensor)
        dist_grad = dist_param.grad.full_tensor()

        check_grad_distance(
            local_grad,
            dist_grad,
            tol_angle=tol_angle,
            tol_norm_abs=tol_norm_abs,
            tol_norm_rel=tol_norm_rel
        )


@torch.no_grad()
def copy_params_local_to_dist(
        local: torch.nn.Module,
        dist: torch.nn.Module
):
    local_params = dict(local.named_parameters())

    for name, dist_param in dist.named_parameters():
        assert name in local_params

        local_param = local_params[name]

        local_data = local_param.data

        assert isinstance(dist_param, DTensor)

        sharded_local_data = distribute_tensor(
            local_data,
            device_mesh=dist_param.device_mesh,
            placements=dist_param.placements
        )

        dist_param.to_local().copy_(sharded_local_data.to_local())
