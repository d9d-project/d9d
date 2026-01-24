import torch
from torch import nn

from d9d.module.block.moe import GroupedLinear

from .config import LoRAParameters


class LoRALinear(nn.Module):
    """
    A LoRA wrapper around a standard PyTorch Linear layer.

    Wraps a base linear layer and adds low-rank adaptation matrices A and B.

    Attributes:
        lora_A: The A matrix (in_features -> r).
        lora_B: The B matrix (r -> out_features).
        base: The original base Linear layer.
        dropout: Scaling dropout layer.
    """

    def __init__(
            self,
            base_layer: nn.Linear,
            params: LoRAParameters
    ):
        """
        Constructs a LoRALinear layer.

        Args:
            base_layer: The original Linear layer to wrap.
            params: LoRA hyperparameters (r, alpha, dropout).

        Raises:
            ValueError: If the base layer has a bias (currently unsupported).
        """

        super().__init__()
        self.lora_A = nn.Linear(
            base_layer.in_features, params.r, bias=False,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype
        )
        self.lora_B = nn.Linear(
            params.r, base_layer.out_features, bias=False,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype
        )
        self.base = base_layer

        if base_layer.bias is not None:
            raise ValueError("LoRA is unsupported with biased linear layers")

        self.dropout: nn.Dropout = nn.Dropout(params.dropout)

        self._scale: float = params.alpha / params.r

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes input tensor, computes base output and LoRA adaptation, and returns the sum.

        Args:
            x: Input tensor.

        Returns:
            The output of base(x) + scale * (B @ A @ dropout(x)).
        """

        base_x = self.base(x)
        adapt_x = self._scale * self.lora_B(self.lora_A(self.dropout(x)))
        return base_x + adapt_x

    @torch.no_grad()
    def merge_with_base_(self) -> nn.Linear:
        """
        Collapse the LoRA weights into the base linear layer.

        Returns:
            The modified base linear layer with updated weights.
        """

        mod = self.base
        mod.weight.data += (self.lora_B.weight.data @ self.lora_A.weight.data) * self._scale
        return mod

    def reset_parameters(self):
        """
        Resets LoRA parameters. A is random, B is zeroed.
        """

        self.lora_A.reset_parameters()
        nn.init.zeros_(self.lora_B.weight)


class LoRAGroupedLinear(nn.Module):
    """
    A LoRA wrapper around a GroupedLinear layer (commonly used in MoE or grouped query attention).

    Attributes:
        lora_A: The A matrix (grouped linear).
        lora_B: The B matrix (grouped linear).
        base: The original base GroupedLinear layer.
        dropout: Scaling dropout layer.
    """

    def __init__(
            self,
            base_layer: GroupedLinear,
            params: LoRAParameters
    ):
        """
        Constructs a LoRAGroupedLinear layer.

        Args:
            base_layer: The original GroupedLinear layer to wrap.
            params: LoRA hyperparameters.
        """

        super().__init__()
        self.lora_A = GroupedLinear(
            base_layer.n_groups, base_layer.in_features, params.r,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype
        )
        self.lora_B = GroupedLinear(
            base_layer.n_groups,
            params.r,
            base_layer.out_features,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype
        )
        self.base = base_layer

        self.dropout = nn.Dropout(params.dropout)

        self._scale = params.alpha / params.r

        self.reset_parameters()

    def forward(self, x: torch.Tensor, x_groups: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass for grouped inputs.

        Args:
            x: Input tensor.
            x_groups: A tensor indicating group indices for each input.

        Returns:
            Combined output of base and LoRA path.
        """

        base_x = self.base(x, x_groups)
        adapt_x = self._scale * self.lora_B(self.lora_A(self.dropout(x), x_groups), x_groups)
        return base_x + adapt_x

    @torch.no_grad()
    def merge_with_base_(self) -> GroupedLinear:
        """
        Collapse the LoRA weights into the base GroupedLinear layer.

        Returns:
            The modified GroupedLinear layer.
        """

        mod = self.base
        mod.weight.data += (torch.bmm(self.lora_A.weight.data, self.lora_B.weight.data)) * self._scale
        return mod

    def reset_parameters(self):
        """
        Resets LoRA parameters. A is random, B is zeroed.
        """

        self.lora_A.reset_parameters()
        nn.init.zeros_(self.lora_B.weight)
