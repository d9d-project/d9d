import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from d9d.module.base import ModuleLateInit

from .grouped_mm import (
    AnyGroupedMmBackendConfig,
    ExpertTokenGroups,
    GroupedMmBackend,
    GroupedMmParameters,
    build_grouped_mm_backend,
)


class GroupedLinear(nn.Module, ModuleLateInit):
    """Linear layer with a separate transformation for each group of tokens, computed with grouped GEMM.

    Each group (expert) has its own weight and processes a variable number of tokens. This is the compute core
    of the Mixture-of-Experts layer.
    """

    def __init__(
        self,
        n_groups: int,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        grouped_mm_backend: AnyGroupedMmBackendConfig | None = None,
    ):
        """Constructs the ``GroupedLinear`` object.

        Args:
            n_groups: Number of groups (experts).
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            device: Target device.
            dtype: Target data type.
            grouped_mm_backend: Grouped GEMM backend. If ``None``, the backend is auto-detected.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_groups, in_features, out_features, device=device, dtype=dtype))

        self.n_groups = n_groups
        self.in_features = in_features
        self.out_features = out_features

        self._grouped_mm_backend = grouped_mm_backend
        self.kernel: GroupedMmBackend = build_grouped_mm_backend(
            GroupedMmParameters(in_features=in_features, out_features=out_features), grouped_mm_backend
        )

        self.reset_parameters()

    def with_features(self, in_features: int, out_features: int) -> "GroupedLinear":
        """Creates a new ``GroupedLinear`` like this one, but with other feature sizes.

        The new layer has the same number of groups, grouped GEMM backend configuration, device and dtype, and
        freshly initialized weights. Layers attached to this one, such as LoRA adapters, use it to run on the same
        backend without knowing which one it is.

        Args:
            in_features: Input hidden size of the new layer.
            out_features: Output hidden size of the new layer.

        Returns:
            The new layer.
        """
        return GroupedLinear(
            self.n_groups,
            in_features,
            out_features,
            grouped_mm_backend=self._grouped_mm_backend,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def forward(self, x: torch.Tensor, x_groups: ExpertTokenGroups) -> torch.Tensor:
        """Performs the grouped matrix multiplication.

        Args:
            x: Input tokens of all groups, with the tokens of each group stored together.
                Shape: ``(num_tokens, in_features)``.
            x_groups: How the tokens are split between groups.

        Returns:
            The output tensor. Shape: ``(num_tokens, out_features)``.
        """
        weight: torch.Tensor = self.weight

        if isinstance(weight, DTensor):
            weight = weight.to_local()

        return self.kernel(x, weight, x_groups)

    def reset_parameters(self):
        """Initializes the weights from ``U(-1 / sqrt(in_features), 1 / sqrt(in_features))``."""
        nn.init.uniform_(self.weight, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))
