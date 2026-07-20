import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch import nn

from ...types import SequencePacking
from ..config import FlashAttention2SdpaBackendConfig, SdpaParameters
from ..protocol import SdpaBackend


class FlashAttention2Sdpa(nn.Module, SdpaBackend):
    """Scaled Dot Product Attention using Flash Attention 2.

    Args:
        config: Backend configuration.
        params: Structural parameters.
    """

    def __init__(self, config: FlashAttention2SdpaBackendConfig, params: SdpaParameters) -> None:
        super().__init__()

        if params.num_sinks is not None:
            raise ValueError("Flash Attention 2 backend does not support learnable sinks (`num_sinks`).")

        self._window_size = params.window_size

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        packing: SequencePacking | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError("Flash Attention 2 does not support setting attention mask explicitly")

        left = self._window_size[0] if self._window_size[0] is not None else -1
        right = self._window_size[1] if self._window_size[1] is not None else -1
        window = (left, right)

        if packing is None:
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
        else:
            if query_states.shape[0] != 1:
                raise ValueError(
                    "Flash Attention 2 sequence packing expects a single packed row (batch dimension "
                    f"of size one), got batch size {query_states.shape[0]}"
                )

            # (1, total, heads, dim) -> (total, heads, dim) for the varlen kernel.
            out = flash_attn_varlen_func(
                query_states.squeeze(0),
                key_states.squeeze(0),
                value_states.squeeze(0),
                cu_seqlens_q=packing.cu_seqlens,
                cu_seqlens_k=packing.cu_seqlens,
                max_seqlen_q=packing.max_seqlen,
                max_seqlen_k=packing.max_seqlen,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=is_causal,
                window_size=window,
            )
            # (total, heads, dim) -> (1, total, heads, dim).
            return out.unsqueeze(0)
