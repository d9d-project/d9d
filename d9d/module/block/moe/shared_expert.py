import torch
from pydantic import BaseModel
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.ffn import SwiGLU


class SharedExpertParameters(BaseModel):
    """
    Configuration parameters for a shared expert.

    Attributes:
        intermediate_size: Dimensionality of the intermediate projection.
        enable_gate: Whether to enable the linear gating mechanism.
    """

    intermediate_size: int
    enable_gate: bool


class SharedSwiGLU(nn.Module, ModuleLateInit):
    """
    A shared expert module using the SwiGLU activation function with an optional gating mechanism.

    Attributes:
        expert: The underlying SwiGLU computation module.
        gate: The optional linear layer used for the gating mechanism.
    """

    def __init__(self, hidden_size: int, params: SharedExpertParameters):
        """
        Constructs the SharedSwiGLU object.

        Args:
            hidden_size: Dimensionality of the hidden state.
            params: Configuration parameters for the shared expert.
        """

        super().__init__()
        self.expert = SwiGLU(hidden_size=hidden_size, intermediate_size=params.intermediate_size)

        if params.enable_gate:
            self.gate = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.gate = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Applies the shared expert computation to the input.

        Args:
            hidden_states: Input tensor to process.

        Returns:
            Output tensor after applying the expert and optional gating.
        """

        x = self.expert(hidden_states)

        if self.gate is not None:
            x = x * torch.sigmoid(self.gate(hidden_states))

        return x

    def reset_parameters(self):
        """
        Resets module parameters.
        """

        self.expert.reset_parameters()

        if self.gate is not None:
            self.gate.reset_parameters()
