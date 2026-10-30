import torch

from d9d.core.autograd import GradDirection
from d9d.kernel.grouped_mm.torch import grouped_mm

from ..alignment import GroupedMmPadding
from ..config import GroupedMmParameters
from ..groups import ExpertTokenGroups
from ..protocol import GroupedMmBackend

# torch grouped_mm checks for 16-byte aligned strides, because its CUTLASS and cuBLASLt kernels load operands with TMA.
_ALIGNMENT_BYTES = 16


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
        out = grouped_mm(
            x, weight, groups.offsets[1:], a_grad_direction=GradDirection.inputs, b_grad_direction=GradDirection.weight
        )
        return self._padding.unpad(out)
