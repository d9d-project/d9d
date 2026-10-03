from typing import Any

import torch
from torch.autograd import Function

from .op import rms_norm_backward, rms_norm_forward


class RMSNormFunction(Function):
    """Autograd function for RMS (root mean square) normalization."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        out, inv_rms = rms_norm_forward(x, weight, eps, zero_centered)
        ctx.inv_rms = inv_rms
        ctx.eps = eps
        ctx.zero_centered = zero_centered
        return out

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override] - torch declares backward with variadic grad_outputs
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x, weight = ctx.saved_tensors
        inv_rms = ctx.inv_rms
        grad_x, grad_weight = rms_norm_backward(grad_output, x, weight, inv_rms, zero_centered=ctx.zero_centered)
        return grad_x, grad_weight, None, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False) -> torch.Tensor:
    """Applies RMS (root mean square) normalization over the last dimension.

    Args:
        x: Input tensor. Shape: ``(..., hidden_size)``.
        weight: Scale for each element of the last dimension. Shape: ``(hidden_size,)``.
        eps: Value added to the mean square for numerical stability.
        zero_centered: If ``True``, the kernel scales by ``weight + 1.0``, so a zero ``weight`` keeps the
            normalized input unchanged.

    Returns:
        The normalized tensor. Shape: ``(..., hidden_size)``.
    """
    return RMSNormFunction.apply(x, weight, eps, zero_centered)
