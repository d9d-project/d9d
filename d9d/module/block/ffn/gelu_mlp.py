import torch
import torch.nn.functional as F
from torch import nn

from d9d.module.base import ModuleLateInit


class GELUMLP(nn.Module, ModuleLateInit):
    """Implements a two-layer Feed-Forward Network (FFN) with GELU activation.

    This module applies `fc2(GELU(fc1(x)))` using the tanh approximation of GELU.
    It corresponds to the standard MLP block used in Vision Transformers.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True):
        """Constructs a GELUMLP object.

        Args:
            hidden_size: The hidden dim size.
            intermediate_size: The intermediate dim size of the FFN.
            bias: Whether to use bias in the linear projections.
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the GELU FFN to the input.

        Args:
            x: Input tensor. Shape: `(..., hidden_size)`.

        Returns:
            Output tensor. Shape: `(..., hidden_size)`.
        """
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

    def reset_parameters(self):
        """Resets module parameters."""
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
