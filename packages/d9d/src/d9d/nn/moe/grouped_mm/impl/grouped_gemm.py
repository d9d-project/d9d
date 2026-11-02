from typing import Any

import torch
from grouped_gemm import backend
from torch.autograd import Function

from d9d.autograd.split_backward import GLOBAL_GRAD_CONTEXT, GradDirection

from ..groups import ExpertTokenGroups
from ..protocol import GroupedMmBackend


class _GroupedGemmFunction(Function):
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
        """Computes the grouped GEMM.

        Each group of consecutive rows of ``a`` is multiplied by its own matrix from ``b``.

        Args:
            ctx: The autograd context.
            a: Left-hand side, with the rows of each group stored together. Shape: ``(num_tokens, in_features)``.
            b: Right-hand side, one matrix per group. Shape: ``(num_groups, in_features, out_features)``, or
                ``(num_groups, out_features, in_features)`` if ``trans_b`` is ``True``.
            batch_sizes: CPU tensor with the number of rows of ``a`` in each group. Must sum to ``num_tokens``.
                Shape: ``(num_groups,)``.
            a_grad_direction: Gradient category for ``a``.
            b_grad_direction: Gradient category for ``b``.
            trans_b: Whether to transpose each matrix of ``b``.

        Returns:
            The result. Shape: ``(num_tokens, out_features)``.
        """
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


class GroupedGemm(GroupedMmBackend):
    """Grouped GEMM through the ``grouped_gemm`` library."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, groups: ExpertTokenGroups) -> torch.Tensor:
        return _GroupedGemmFunction.apply(
            x,
            weight,
            groups.host_counts(),
            a_grad_direction=GradDirection.inputs,
            b_grad_direction=GradDirection.weight,
            trans_b=False,
        )
