from typing import Any, cast

import torch
from quack.gemm_interface import gemm
from torch.autograd import Function

from d9d.autograd.split_backward import GLOBAL_GRAD_CONTEXT, GradDirection

from ..alignment import GroupedMmPadding
from ..config import GroupedMmParameters
from ..groups import ExpertTokenGroups
from ..protocol import GroupedMmBackend

# QuACK GEMMs need the contiguous dimension of every operand to be 16-byte aligned, because they load it with TMA.
_ALIGNMENT_BYTES = 16


class _QuackGroupedMmFunction(Function):
    """Autograd function for QuACK variable-length GEMMs with explicit gradient control."""

    @staticmethod
    def forward(
        ctx: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        cu_rows: torch.Tensor,
        a_grad_direction: GradDirection | None,
        b_grad_direction: GradDirection | None,
    ) -> torch.Tensor:
        """Computes the grouped GEMM.

        Args:
            ctx: The autograd context.
            a: Rows of all groups, sorted by group. Shape: ``(total_rows, in_features)``.
            b: Per-group weights. Shape: ``(n_groups, in_features, out_features)``.
            cu_rows: Row offsets of the groups with a leading zero, on the device. int32, shape ``(n_groups + 1,)``.
            a_grad_direction: Gradient category for ``a``.
            b_grad_direction: Gradient category for ``b``.

        Returns:
            The result. Shape: ``(total_rows, out_features)``.
        """
        ctx.save_for_backward(a, b, cu_rows)
        ctx.a_grad_direction = a_grad_direction
        ctx.b_grad_direction = b_grad_direction
        # Plain bf16 operands produce a plain tensor. QuACK returns BlockScaledOperand only for quantized outputs.
        # Untuned, because QuACK re-tunes for every new token count.
        return cast(torch.Tensor, gemm(a, b, cu_seqlens_m=cu_rows, tuned=False))

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override] - torch declares backward with variadic grad_outputs
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        grad = grad.contiguous()
        a, b, cu_rows = ctx.saved_tensors

        a_grad = None
        if ctx.needs_input_grad[0] and GLOBAL_GRAD_CONTEXT.check_direction(ctx.a_grad_direction):
            a_grad = cast(torch.Tensor, gemm(grad, b.mT, cu_seqlens_m=cu_rows, tuned=False))

        b_grad = None
        if ctx.needs_input_grad[1] and GLOBAL_GRAD_CONTEXT.check_direction(ctx.b_grad_direction):
            # Groups run along the contracted dimension. Experts without rows get zeros.
            b_grad = torch.empty_like(b)
            gemm(a.T, grad, out=b_grad, cu_seqlens_k=cu_rows, tuned=False)

        return a_grad, b_grad, None, None, None


class QuackGroupedMm(GroupedMmBackend):
    """Grouped GEMM through QuACK variable-length GEMMs."""

    def __init__(self, params: GroupedMmParameters) -> None:
        """Constructs the ``QuackGroupedMm`` object.

        Args:
            params: Structural layer parameters.
        """
        self._padding = GroupedMmPadding(params, alignment_bytes=_ALIGNMENT_BYTES)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, groups: ExpertTokenGroups) -> torch.Tensor:
        x, weight = self._padding.pad(x, weight)
        out = _QuackGroupedMmFunction.apply(
            x,
            weight,
            groups.offsets,
            a_grad_direction=GradDirection.inputs,
            b_grad_direction=GradDirection.weight,
        )
        return self._padding.unpad(out)
