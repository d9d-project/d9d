import torch
from torch import nn

from d9d.module.block.moe import GroupedLinear

from .config import LoRAParameters


class LoRALinear(nn.Module):
    """Wrapper that adds low-rank adaptation matrices A and B to an ``nn.Linear`` layer.

    Attributes:
        lora_A: The A matrix (``in_features -> r``).
        lora_B: The B matrix (``r -> out_features``).
        base: The original ``nn.Linear`` layer.
        dropout: Dropout applied to the input of the LoRA path.
    """

    def __init__(self, base_layer: nn.Linear, params: LoRAParameters):
        """Constructs the ``LoRALinear`` object.

        Args:
            base_layer: The original ``nn.Linear`` layer to wrap.
            params: LoRA hyperparameters.

        Raises:
            ValueError: If ``base_layer`` has a bias.
        """
        super().__init__()
        self.lora_A = nn.Linear(
            base_layer.in_features, params.r, bias=False, device=base_layer.weight.device, dtype=base_layer.weight.dtype
        )
        self.lora_B = nn.Linear(
            params.r,
            base_layer.out_features,
            bias=False,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        self.base = base_layer

        if base_layer.bias is not None:
            raise ValueError(
                "LoRA does not support linear layers with a bias. Apply it only to layers with bias=False."
            )

        self.dropout: nn.Dropout = nn.Dropout(params.dropout)

        self._scale: float = params.alpha / params.r

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the base output plus the scaled LoRA update.

        Args:
            x: Input tensor.

        Returns:
            ``base(x) + alpha / r * lora_B(lora_A(dropout(x)))``.
        """
        base_x = self.base(x)
        adapt_x = self._scale * self.lora_B(self.lora_A(self.dropout(x)))
        return base_x + adapt_x

    @torch.no_grad()
    def merge_with_base_(self) -> nn.Linear:
        """Merges the LoRA weights into the base linear layer in place.

        Returns:
            The base linear layer with updated weights.
        """
        mod = self.base
        mod.weight.data += (self.lora_B.weight.data @ self.lora_A.weight.data) * self._scale
        return mod

    def reset_parameters(self):
        """Resets the LoRA parameters.

        ``lora_A`` gets a random initialization and ``lora_B`` is zeroed, so the LoRA update starts at zero.
        """
        self.lora_A.reset_parameters()
        nn.init.zeros_(self.lora_B.weight)


class LoRAGroupedLinear(nn.Module):
    """Wrapper that adds low-rank adaptation matrices A and B to a ``GroupedLinear`` layer, as used by MoE experts.

    Attributes:
        lora_A: The A matrix (``in_features -> r`` for each group).
        lora_B: The B matrix (``r -> out_features`` for each group).
        base: The original ``GroupedLinear`` layer.
        dropout: Dropout applied to the input of the LoRA path.
    """

    def __init__(self, base_layer: GroupedLinear, params: LoRAParameters):
        """Constructs the ``LoRAGroupedLinear`` object.

        Args:
            base_layer: The original ``GroupedLinear`` layer to wrap.
            params: LoRA hyperparameters.
        """
        super().__init__()
        self.lora_A = GroupedLinear(
            base_layer.n_groups,
            base_layer.in_features,
            params.r,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        self.lora_B = GroupedLinear(
            base_layer.n_groups,
            params.r,
            base_layer.out_features,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        self.base = base_layer

        self.dropout = nn.Dropout(params.dropout)

        self._scale = params.alpha / params.r

        self.reset_parameters()

    def forward(self, x: torch.Tensor, x_groups: torch.Tensor) -> torch.Tensor:
        """Computes the base output plus the scaled LoRA update for grouped inputs.

        Args:
            x: Input tokens of all groups, with the tokens of each group stored together.
                Shape: ``(num_tokens, in_features)``.
            x_groups: CPU tensor with the number of tokens in each group. Shape: ``(n_groups,)``.

        Returns:
            The sum of the base and LoRA outputs. Shape: ``(num_tokens, out_features)``.
        """
        base_x = self.base(x, x_groups)
        adapt_x = self._scale * self.lora_B(self.lora_A(self.dropout(x), x_groups), x_groups)
        return base_x + adapt_x

    @torch.no_grad()
    def merge_with_base_(self) -> GroupedLinear:
        """Merges the LoRA weights into the base ``GroupedLinear`` layer in place.

        Returns:
            The base ``GroupedLinear`` layer with updated weights.
        """
        mod = self.base
        mod.weight.data += (torch.bmm(self.lora_A.weight.data, self.lora_B.weight.data)) * self._scale
        return mod

    def reset_parameters(self):
        """Resets the LoRA parameters.

        ``lora_A`` gets a random initialization and ``lora_B`` is zeroed, so the LoRA update starts at zero.
        """
        self.lora_A.reset_parameters()
        nn.init.zeros_(self.lora_B.weight)
