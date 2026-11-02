from typing import Any, cast

import torch
from quack.gemm_interface import gemm
from torch.autograd import Function

from d9d.autograd.split_backward import GradDirection, needs_input_grad

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
    ) -> torch.Tensor:
        """Computes the grouped GEMM.

        Args:
            ctx: The autograd context.
            a: Rows of all groups, sorted by group. Shape: ``(total_rows, in_features)``.
            b: Per-group weights. Shape: ``(n_groups, in_features, out_features)``.
            cu_rows: Row offsets of the groups with a leading zero, on the device. int32, shape ``(n_groups + 1,)``.

        Returns:
            The result. Shape: ``(total_rows, out_features)``.
        """
        ctx.save_for_backward(a, b, cu_rows)
        # Plain bf16 operands produce a plain tensor. QuACK returns BlockScaledOperand only for quantized outputs.
        # Untuned, because QuACK re-tunes for every new token count.
        return cast(torch.Tensor, gemm(a, b, cu_seqlens_m=cu_rows, tuned=False))

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override] - torch declares backward with variadic grad_outputs
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        grad = grad.contiguous()
        a, b, cu_rows = ctx.saved_tensors

        a_grad = None
        if needs_input_grad(ctx, 0, GradDirection.inputs):
            a_grad = cast(torch.Tensor, gemm(grad, b.mT, cu_seqlens_m=cu_rows, tuned=False))

        b_grad = None
        if needs_input_grad(ctx, 1, GradDirection.weight):
            # Groups run along the contracted dimension. Experts without rows get zeros.
            b_grad = torch.empty_like(b)
            gemm(a.T, grad, out=b_grad, cu_seqlens_k=cu_rows, tuned=False)

        return a_grad, b_grad, None


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
        out = _QuackGroupedMmFunction.apply(x, weight, groups.offsets)
        return self._padding.unpad(out)
