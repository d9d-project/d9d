from typing import Any

import torch
from torch.autograd import Function

from .op import silu_mul_backward, silu_mul_forward


class SiLUMulFunction(Function):
    """
    Autograd function for the fused silu(x)*y operation.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, y)
        return silu_mul_forward(x, y)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[invalid-method-override]
        x, y = ctx.saved_tensors
        return silu_mul_backward(grad_output, x, y)


def silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Applies the SiLU multiplication operation: SiLU(x) * y.

    Args:
        x: Input tensor x.
        y: Input tensor y.

    Returns:
        The resulting tensor of the same shape as inputs.
    """
    return SiLUMulFunction.apply(x, y)
