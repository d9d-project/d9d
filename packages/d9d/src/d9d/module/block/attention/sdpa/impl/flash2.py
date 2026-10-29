import torch
from flash_attn import flash_attn_func
from torch import nn

from ..config import FlashAttention2SdpaBackendConfig, SdpaParameters
from ..protocol import SdpaBackend


class FlashAttention2Sdpa(nn.Module, SdpaBackend):
    """Scaled dot-product attention that uses FlashAttention 2."""

    def __init__(self, config: FlashAttention2SdpaBackendConfig, params: SdpaParameters) -> None:
        """Constructs the ``FlashAttention2Sdpa`` object.

        Args:
            config: Backend configuration.
            params: Structural layer parameters.

        Raises:
            ValueError: If ``params`` enables learnable sinks.
        """
        super().__init__()

        if params.num_sinks is not None:
            raise ValueError(
                f"The FlashAttention 2 backend does not support learnable sinks, but num_sinks ({params.num_sinks}) "
                f"is set. Use the FlashAttention 4 or eager backend."
            )

        self._window_size = params.window_size

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError(
                "The FlashAttention 2 backend does not support an explicit attention mask. "
                "Pass attention_mask=None or use the PyTorch SDPA or eager backend."
            )

        left = self._window_size[0] if self._window_size[0] is not None else -1
        right = self._window_size[1] if self._window_size[1] is not None else -1
        window = (left, right)

        out = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=is_causal,
            window_size=window,
        )

        return out
