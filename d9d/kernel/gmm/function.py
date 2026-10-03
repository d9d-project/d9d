from typing import Any

import torch
from grouped_gemm import backend
from torch.autograd import Function

from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection


class GroupedGemm(Function):
    """Autograd function for grouped GEMM (general matrix multiplication) with explicit gradient control."""

    @staticmethod
    def forward(
        ctx: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        a_grad_direction: GradDirection | None,
        b_grad_direction: GradDirection | None,
        trans_b: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.a_grad_direction = a_grad_direction
        ctx.b_grad_direction = b_grad_direction
        ctx.trans_b = trans_b
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override] - torch declares backward with variadic grad_outputs
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        compute_a = GLOBAL_GRAD_CONTEXT.check_direction(ctx.a_grad_direction)
        compute_b = GLOBAL_GRAD_CONTEXT.check_direction(ctx.b_grad_direction)

        a_grad = None
        if ctx.needs_input_grad[0] and compute_a:
            a_grad = backend.gmm(grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        b_grad = None
        if ctx.needs_input_grad[1] and compute_b:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            b_grad = backend.gmm(lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return a_grad, b_grad, None, None, None, None


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    a_grad_direction: GradDirection | None,
    b_grad_direction: GradDirection | None,
    trans_b: bool = False,
) -> torch.Tensor:
    """Computes a grouped GEMM (general matrix multiplication) with explicit gradient control.

    Each group of consecutive rows of ``a`` is multiplied by its own matrix from ``b``.

    Args:
        a: Left-hand side, with the rows of each group stored together. Shape: ``(num_tokens, in_features)``.
        b: Right-hand side, one matrix per group. Shape: ``(num_groups, in_features, out_features)``, or
            ``(num_groups, out_features, in_features)`` if ``trans_b`` is ``True``.
        batch_sizes: CPU tensor with the number of rows of ``a`` in each group. Must sum to ``num_tokens``.
            Shape: ``(num_groups,)``.
        a_grad_direction: Gradient category for ``a`` (e.g. ``GradDirection.inputs``).
        b_grad_direction: Gradient category for ``b`` (e.g. ``GradDirection.weight``).
        trans_b: Whether to transpose each matrix of ``b``.

    Returns:
        The result. Shape: ``(num_tokens, out_features)``.
    """
    return GroupedGemm.apply(a, b, batch_sizes, a_grad_direction, b_grad_direction, trans_b)
