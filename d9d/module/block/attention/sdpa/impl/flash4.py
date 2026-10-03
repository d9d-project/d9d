import torch
import torch.nn.functional as F
from flash_attn.cute import flash_attn_func
from torch import nn

from ..config import FlashAttention4SdpaBackendConfig, SdpaParameters
from ..protocol import SdpaBackend

# FA4's backward preprocess kernel requires head_dim to be a multiple of this value.
# Other head dims hit a CuTe predicate-shape bug.
_FA4_HDIM_ALIGN = 32


class FlashAttention4Sdpa(nn.Module, SdpaBackend):
    """Scaled dot-product attention that uses FlashAttention 4.

    If ``num_sinks`` is set, a learnable per-head sink logit is added to the softmax denominator.
    The sink absorbs part of the attention mass.
    """

    def __init__(self, config: FlashAttention4SdpaBackendConfig, params: SdpaParameters) -> None:
        """Constructs the ``FlashAttention4Sdpa`` object.

        Args:
            config: Backend configuration.
            params: Structural layer parameters.
        """
        super().__init__()

        self.sinks = nn.Parameter(torch.zeros(params.num_sinks)) if params.num_sinks is not None else None

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
                "The FlashAttention 4 backend does not support an explicit attention mask. "
                "Pass attention_mask=None or use the PyTorch SDPA or eager backend."
            )

        # Pad head_dim to a multiple of _FA4_HDIM_ALIGN. Zero-padding does not change the result:
        # extra dims add 0 to QK^T and to the weighted sum over V.
        head_dim = query_states.shape[-1]
        pad = (-head_dim) % _FA4_HDIM_ALIGN
        if pad:
            query_states = F.pad(query_states, (0, pad))
            key_states = F.pad(key_states, (0, pad))
            value_states = F.pad(value_states, (0, pad))

        out, *_ = flash_attn_func(
            query_states,
            key_states,
            value_states,
            softmax_scale=scale,
            causal=is_causal,
            window_size=self._window_size,
            learnable_sink=self.sinks,
        )

        if pad:
            out = out[..., :head_dim]

        return out
