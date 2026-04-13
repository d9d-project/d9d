from typing import Any

import torch
from torch.autograd import Function

from .op import rms_norm_backward, rms_norm_forward


class RMSNormFunction(Function):
    """Custom PyTorch autograd function for Root Mean Square (RMS) normalization."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        out, inv_rms = rms_norm_forward(x, weight, eps, zero_centered)
        ctx.inv_rms = inv_rms
        ctx.eps = eps
        ctx.zero_centered = zero_centered
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:  # type: ignore[invalid-method-override]
        x, weight = ctx.saved_tensors
        inv_rms = ctx.inv_rms
        grad_x, grad_weight = rms_norm_backward(grad_output, x, weight, inv_rms, zero_centered=ctx.zero_centered)
        return grad_x, grad_weight, None, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False) -> torch.Tensor:
    """Applies Root Mean Square (RMS) normalization to the input tensor.

    Args:
        x: Input tensor to normalize.
        weight: Learnable scaling parameters.
        eps: A small value added to the variance for numerical stability.
        zero_centered: If True, the learned weights are computationally centered around zero
            by artificially offsetting them by 1.0.

    Returns:
        The normalized output tensor.
    """
    return RMSNormFunction.apply(x, weight, eps, zero_centered)
