from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd import Function

from d9d.autograd.split_backward import GradDirection, needs_input_grad

from ..alignment import GroupedMmPadding
from ..config import GroupedMmParameters
from ..groups import ExpertTokenGroups
from ..protocol import GroupedMmBackend

# torch grouped_mm checks for 16-byte aligned strides, because its CUTLASS and cuBLASLt kernels load operands with TMA.
_ALIGNMENT_BYTES = 16


class _TorchGroupedMmFunction(Function):
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
    ) -> torch.Tensor:
        """Computes the grouped GEMM.

        Args:
            ctx: The autograd context.
            a: Rows of all groups, sorted by group. Shape: ``(total_rows, in_features)``.
            b: Per-group weights. Shape: ``(n_groups, in_features, out_features)``.
            group_ends: Cumulative row count at the end of each group, on the device. int32, shape ``(n_groups,)``.

        Returns:
            The result. Shape: ``(total_rows, out_features)``.
        """
        ctx.save_for_backward(a, b, group_ends)
        return F.grouped_mm(a, b, offs=group_ends)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override] - torch declares backward with variadic grad_outputs
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        grad = grad.contiguous()
        a, b, group_ends = ctx.saved_tensors

        a_grad = None
        if needs_input_grad(ctx, 0, GradDirection.inputs):
            a_grad = F.grouped_mm(grad, b.transpose(-2, -1), offs=group_ends)

        b_grad = None
        if needs_input_grad(ctx, 1, GradDirection.weight):
            # In the 2D x 2D form, groups run along the contracted dimension.
            b_grad = F.grouped_mm(a.t(), grad, offs=group_ends)

        return a_grad, b_grad, None


class TorchGroupedMm(GroupedMmBackend):
    """Grouped GEMM through ``torch.nn.functional.grouped_mm``."""

    def __init__(self, params: GroupedMmParameters) -> None:
        """Constructs the ``TorchGroupedMm`` object.

        Args:
            params: Structural layer parameters.
        """
        self._padding = GroupedMmPadding(params, alignment_bytes=_ALIGNMENT_BYTES)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, groups: ExpertTokenGroups) -> torch.Tensor:
        x, weight = self._padding.pad(x, weight)
        out = _TorchGroupedMmFunction.apply(x, weight, groups.offsets[1:])
        return self._padding.unpad(out)
