import torch

from d9d.core.autograd import GradDirection
from d9d.kernel.grouped_mm.quack import grouped_mm

from ..alignment import GroupedMmPadding
from ..config import GroupedMmParameters
from ..groups import ExpertTokenGroups
from ..protocol import GroupedMmBackend

# QuACK GEMMs need the contiguous dimension of every operand to be 16-byte aligned, because they load it with TMA.
_ALIGNMENT_BYTES = 16


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
        out = grouped_mm(
            x,
            weight,
            groups.offsets,
            a_grad_direction=GradDirection.inputs,
            b_grad_direction=GradDirection.weight,
        )
        return self._padding.unpad(out)
