import torch
from torch import nn

from d9d.kernel.swiglu import silu_mul
from d9d.module.base import ModuleLateInit


class SwiGLU(nn.Module, ModuleLateInit):
    """SwiGLU feed-forward network (FFN).

    Computes ``down(SiLU(gate(x)) * up(x))``. This is the standard MLP block of architectures like LLaMA.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        """Constructs the ``SwiGLU`` object.

        Args:
            hidden_size: Hidden size.
            intermediate_size: Intermediate size of the FFN.
            bias: Whether to use bias in the linear projections.
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SwiGLU FFN to the input.

        Args:
            x: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor. Shape: ``(batch, seq_len, hidden_size)``.
        """
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))

    def reset_parameters(self):
        """Resets module parameters."""
        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()
