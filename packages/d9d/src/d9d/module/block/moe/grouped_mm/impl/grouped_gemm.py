import torch

from d9d.core.autograd import GradDirection
from d9d.kernel.grouped_mm.nv import grouped_mm

from ..groups import ExpertTokenGroups
from ..protocol import GroupedMmBackend


class GroupedGemm(GroupedMmBackend):
    """Grouped GEMM through the ``grouped_gemm`` library."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, groups: ExpertTokenGroups) -> torch.Tensor:
        return grouped_mm(
            x,
            weight,
            groups.host_counts(),
            a_grad_direction=GradDirection.inputs,
            b_grad_direction=GradDirection.weight,
        )
