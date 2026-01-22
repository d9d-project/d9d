import torch
from torch import Tensor


def check_grad(my_grad: Tensor, hf_grad: torch.Tensor, is_dist: bool, atol: float, rtol: float):
    if is_dist:
        my_grad = my_grad.full_tensor()

    assert torch.allclose(my_grad, hf_grad, atol=atol, rtol=rtol)


def check_grad_distance(
        my_grad: torch.Tensor,
        hf_grad: torch.Tensor,
        is_dist: bool,
        tol_angle: float = 0.05,
        tol_norm_abs: float = 3e-3,
        tol_norm_rel: float = 0.1

):
    if is_dist:
        my_grad = my_grad.full_tensor()

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
