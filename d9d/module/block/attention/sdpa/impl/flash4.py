import torch
import torch.nn.functional as F
from torch import nn

from d9d.kernel.flash_attn import flash_attn_func

from ..config import FlashAttention4SdpaBackendConfig, SdpaParameters
from ..protocol import SdpaBackend

# FA4's backward preprocess kernel requires head_dim to be a multiple of this
# value; non-aligned dims trigger a CUTE predicate-shape bug.
_FA4_HDIM_ALIGN = 32


class FlashAttention4Sdpa(nn.Module, SdpaBackend):
    """Scaled Dot Product Attention using Flash Attention 4.

    When ``num_sinks`` is provided, a learnable per-head sink logit is added
    to the softmax denominator (attention-sink mechanism).  This lets a
    fraction of attention mass be absorbed by the sink, effectively
    soft-gating the output without materializing an extra KV column.

    Args:
        config: Backend configuration.
        params: Structural parameters.
    """

    def __init__(self, config: FlashAttention4SdpaBackendConfig, params: SdpaParameters) -> None:
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
            raise ValueError("Flash Attention 4 does not support setting attention mask explicitly")

        # Pad head_dim to the next multiple of _FA4_HDIM_ALIGN so that FA4's
        # backward preprocess kernel avoids the broken predicate-masking path.
        # Zero-padding is transparent: extra dims contribute 0 to QK^T and to
        # the weighted sum over V.
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
