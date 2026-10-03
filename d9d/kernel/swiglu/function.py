from typing import Any

import torch
from torch.autograd import Function

from .op import silu_mul_backward, silu_mul_forward


class SiLUMulFunction(Function):
    """Autograd function for the fused ``silu(x) * y`` operation."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, y)
        return silu_mul_forward(x, y)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override] - torch declares backward with variadic grad_outputs
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = ctx.saved_tensors
        return silu_mul_backward(grad_output, x, y)


def silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes ``silu(x) * y`` in a fused Triton kernel.

    Args:
        x: Input passed through SiLU.
        y: Input multiplied by ``silu(x)``. Must have the same shape and device as ``x``.

    Returns:
        The result, with the same shape as the inputs.

    Raises:
        ValueError: If ``x`` and ``y`` differ in shape or device.
    """
    return SiLUMulFunction.apply(x, y)
