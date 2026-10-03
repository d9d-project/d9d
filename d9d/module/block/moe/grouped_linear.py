import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from d9d.core.autograd import GradDirection
from d9d.kernel.gmm import gmm
from d9d.module.base import ModuleLateInit


class GroupedLinear(nn.Module, ModuleLateInit):
    """Linear layer with a separate transformation for each group of tokens, computed with grouped GEMM.

    Each group (expert) has its own weight and processes a variable number of tokens. This is the compute core
    of the Mixture-of-Experts layer. Requires the ``d9d[moe]`` extra.
    """

    def __init__(
        self,
        n_groups: int,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Constructs the ``GroupedLinear`` object.

        Args:
            n_groups: Number of groups (experts).
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            device: Target device.
            dtype: Target data type.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_groups, in_features, out_features, device=device, dtype=dtype))

        self.n_groups = n_groups
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def forward(self, x: torch.Tensor, x_groups: torch.Tensor) -> torch.Tensor:
        """Performs the grouped matrix multiplication.

        Args:
            x: Input tokens of all groups, with the tokens of each group stored together.
                Shape: ``(num_tokens, in_features)``.
            x_groups: CPU tensor with the number of tokens in each group. Must sum to ``num_tokens``.
                Shape: ``(n_groups,)``.

        Returns:
            The output tensor. Shape: ``(num_tokens, out_features)``.
        """
        weight: torch.Tensor = self.weight

        if isinstance(weight, DTensor):
            weight = weight.to_local()

        return gmm(x, weight, x_groups, a_grad_direction=GradDirection.inputs, b_grad_direction=GradDirection.weight)

    def reset_parameters(self):
        """Initializes the weights from ``U(-1 / sqrt(in_features), 1 / sqrt(in_features))``."""
        nn.init.uniform_(self.weight, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))
