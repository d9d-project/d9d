import torch
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from torch import nn

from d9d.module.base import ModuleLateInit

# TODO: migrate from liger to custom silu mul


class SwiGLU(nn.Module, ModuleLateInit):
    """
    Implements the SwiGLU Feed-Forward Network (FFN).

    This module applies the gated activation function: `down(SiLU(gate(x)) * up(x))`.
    It corresponds to the standard MLP block used in architectures like LLaMA.
    """

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int
    ):
        """
        Constructs a SwiGLU object.

        Args:
            hidden_size: The hidden dim size.
            intermediate_size: The intermediate dim size of the FFN.
        """

        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the SwiGLU FFN to the input.

        Args:
            x: Input tensor. Shape: `(batch_size, seq_len, hidden_dim)`.

        Returns:
            Output tensor. Shape: `(batch_size, seq_len, hidden_dim)`.
        """

        return self.down_proj(
            LigerSiLUMulFunction.apply(
                self.gate_proj(x),
                self.up_proj(x)
            )
        )

    def reset_parameters(self):
        """Resets module parameters."""

        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()
