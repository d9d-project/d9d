from typing import Protocol

import torch

from .groups import ExpertTokenGroups


class GroupedMmBackend(Protocol):
    """Protocol for grouped matrix multiplication backends used by ``GroupedLinear``."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, groups: ExpertTokenGroups) -> torch.Tensor:
        """Multiplies the rows of every group by the weight of that group.

        Args:
            x: Rows of all groups, sorted by group. Shape: ``(total_rows, in_features)``.
            weight: Per-group weights. Shape: ``(n_groups, in_features, out_features)``.
            groups: How the rows are split between groups.

        Returns:
            The result. Shape: ``(total_rows, out_features)``.
        """
        ...
