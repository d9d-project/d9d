import torch

from ..groups import ExpertTokenGroups
from ..protocol import GroupedMmBackend


class EagerGroupedMm(GroupedMmBackend):
    """Grouped GEMM as one ``matmul`` per group: a portable reference."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, groups: ExpertTokenGroups) -> torch.Tensor:
        rows = torch.split(x, groups.host_counts().tolist())
        return torch.cat([group_rows @ group_weight for group_rows, group_weight in zip(rows, weight, strict=True)])
