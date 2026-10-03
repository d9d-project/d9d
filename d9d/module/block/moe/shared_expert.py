import torch
from pydantic import BaseModel
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.ffn import SwiGLU


class SharedExpertParameters(BaseModel):
    """Configuration for a shared expert.

    Attributes:
        intermediate_size: Intermediate size of the SwiGLU FFN.
        enable_gate: Whether to scale the output with a learned sigmoid gate.
    """

    intermediate_size: int
    enable_gate: bool


class SharedSwiGLU(nn.Module, ModuleLateInit):
    """Shared expert: a SwiGLU FFN with an optional sigmoid output gate."""

    def __init__(self, hidden_size: int, params: SharedExpertParameters):
        """Constructs the ``SharedSwiGLU`` object.

        Args:
            hidden_size: Hidden size.
            params: Shared expert configuration.
        """
        super().__init__()
        self.expert = SwiGLU(hidden_size=hidden_size, intermediate_size=params.intermediate_size)

        if params.enable_gate:
            self.gate = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.gate = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Applies the shared expert computation to the input.

        Args:
            hidden_states: Input tensor. Shape: ``(num_tokens, hidden_size)``.

        Returns:
            Output tensor. Shape: ``(num_tokens, hidden_size)``.
        """
        x = self.expert(hidden_states)

        if self.gate is not None:
            x = x * torch.sigmoid(self.gate(hidden_states))

        return x

    def reset_parameters(self):
        """Resets module parameters."""
        self.expert.reset_parameters()

        if self.gate is not None:
            self.gate.reset_parameters()
