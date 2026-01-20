import math

import torch
from grouped_gemm.ops import gmm
from torch import nn
from torch.distributed.tensor import DTensor

from d9d.module.base import ModuleLateInit


class GroupedLinear(nn.Module, ModuleLateInit):
    """
    Applies a linear transformation using Grouped GEMM (Generalized Matrix Multiplication).

    This module allows efficient execution of multiple linear layers (experts) in parallel, where each expert
    processes a variable number of tokens.
    It is the computational core of the Mixture-of-Experts layer.
    """

    def __init__(
            self,
            n_groups: int,
            in_features: int,
            out_features: int,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None
    ):
        """
        Constructs the GroupedLinear layer.

        Args:
            n_groups: Number of groups (experts).
            in_features: Input hidden size.
            out_features: Output hidden size.
            device: Target device.
            dtype: Target data type.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_groups, in_features, out_features,
                                               device=device, dtype=dtype))

        self.n_groups = n_groups
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def forward(self, x: torch.Tensor, x_groups: torch.Tensor) -> torch.Tensor:
        """
        Performs the grouped matrix multiplication.

        Args:
            x: Flattened input tensor containing tokens for all groups.
                Shape: `(total_tokens, in_features)`.
            x_groups: CPU Tensor indicating the number of tokens assigned to each group.
                Must sum to `total_tokens`. Shape: `(n_groups,)`.

        Returns:
            The output tensor. Shape: `(total_tokens, out_features)`.
        """

        weight: torch.Tensor = self.weight

        if isinstance(weight, DTensor):
            weight = weight.to_local()

        return gmm(x, weight, x_groups)

    def reset_parameters(self):
        """Initializes weights using a uniform distribution based on input features."""
        nn.init.uniform_(self.weight, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))
