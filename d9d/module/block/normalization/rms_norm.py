import torch
from torch import nn

from d9d.kernel.normalization import rms_norm
from d9d.module.base import ModuleLateInit


class RMSNorm(nn.Module, ModuleLateInit):
    """Root Mean Square Layer Normalization (RMSNorm) layer.

    Normalizes the input over its last dimension by its root mean square and applies learnable scaling
    weights. The weights can optionally be zero-centered.

    References:
        [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, zero_centered: bool = False) -> None:
        """Constructs the ``RMSNorm`` object.

        Args:
            hidden_size: Size of the normalized last dimension.
            eps: Small value added to the variance for numerical stability.
            zero_centered: If ``True``, the scaling weights are initialized to 0 and offset by 1 during
                computation. Otherwise, they are initialized to 1.
        """
        super().__init__()
        self._eps = eps
        self._zero_centered = zero_centered

        self.weight = nn.Parameter(torch.empty((hidden_size,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMSNorm to the input.

        Args:
            x: Input tensor. Shape: ``(..., hidden_size)``.

        Returns:
            The normalized tensor. Shape: ``(..., hidden_size)``.
        """
        return rms_norm(x, self.weight, eps=self._eps, zero_centered=self._zero_centered)

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        if self._zero_centered:
            nn.init.zeros_(self.weight)
        else:
            nn.init.ones_(self.weight)
