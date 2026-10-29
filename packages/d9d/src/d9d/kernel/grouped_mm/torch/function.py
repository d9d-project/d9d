from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd import Function

from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection


class TorchGroupedMmFunction(Function):
    """Autograd function for ``torch.nn.functional.grouped_mm`` with explicit gradient control.

    The native backward of ``grouped_mm`` rejects gradients with zero or unaligned strides (e.g. from ``out.sum()``),
    so this function makes the incoming gradient contiguous first.
    """

    @staticmethod
    def forward(
        ctx: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        group_ends: torch.Tensor,
        a_grad_direction: GradDirection | None,
        b_grad_direction: GradDirection | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(a, b, group_ends)
        ctx.a_grad_direction = a_grad_direction
        ctx.b_grad_direction = b_grad_direction
        return F.grouped_mm(a, b, offs=group_ends)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override] - torch declares backward with variadic grad_outputs
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        grad = grad.contiguous()
        a, b, group_ends = ctx.saved_tensors

        a_grad = None
        if ctx.needs_input_grad[0] and GLOBAL_GRAD_CONTEXT.check_direction(ctx.a_grad_direction):
            a_grad = F.grouped_mm(grad, b.transpose(-2, -1), offs=group_ends)

        b_grad = None
        if ctx.needs_input_grad[1] and GLOBAL_GRAD_CONTEXT.check_direction(ctx.b_grad_direction):
            # In the 2D x 2D form, groups run along the contracted dimension.
            b_grad = F.grouped_mm(a.t(), grad, offs=group_ends)

        return a_grad, b_grad, None, None, None


def grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    group_ends: torch.Tensor,
    a_grad_direction: GradDirection | None,
    b_grad_direction: GradDirection | None,
) -> torch.Tensor:
    """Computes a grouped GEMM with ``torch.nn.functional.grouped_mm``, with explicit gradient control.

    Args:
        a: Rows of all groups, sorted by group. Shape: ``(total_rows, in_features)``.
        b: Per-group weights. Shape: ``(n_groups, in_features, out_features)``.
        group_ends: Cumulative row count at the end of each group, on the device. int32, shape ``(n_groups,)``.
        a_grad_direction: Gradient category for ``a`` (e.g. ``GradDirection.inputs``).
        b_grad_direction: Gradient category for ``b`` (e.g. ``GradDirection.weight``).

    Returns:
        The result. Shape: ``(total_rows, out_features)``.
    """
    return TorchGroupedMmFunction.apply(a, b, group_ends, a_grad_direction, b_grad_direction)
