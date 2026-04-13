import torch
from torch import nn

from d9d.kernel.normalization import rms_norm
from d9d.module.base import ModuleLateInit


class RMSNorm(nn.Module, ModuleLateInit):
    """
    Implements Root Mean Square (RMS) Normalization.

    This module normalizes the input tensor across its last dimension using the root mean square
    statistic, applying learnable scaling weights. It can optionally use zero-centered weights.

    References:
        [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, zero_centered: bool = False) -> None:
        """
        Constructs an RMSNorm object.

        Args:
            hidden_size: The dimensionality of the hidden size to normalize.
            eps: A small value added to the variance for numerical stability to prevent division by zero.
            zero_centered: If True, the scaling weights are initialized to 0.0 and
                implicitly offset by 1.0 during computation. Otherwise, they are initialized to 1.0.
        """
        super().__init__()
        self._eps = eps
        self._zero_centered = zero_centered

        self.weight = nn.Parameter(torch.empty((hidden_size,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x: Input tensor to be normalized. The normalization is applied over the last dimension.

        Returns:
            The normalized tensor with the same shape as the input.
        """
        return rms_norm(x, self.weight, eps=self._eps, zero_centered=self._zero_centered)

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        if self._zero_centered:
            nn.init.zeros_(self.weight)
        else:
            nn.init.ones_(self.weight)
