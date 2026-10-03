import torch
from flash_attn import flash_attn_varlen_func
from torch import nn

from ..config import FlashAttention2SdpaBackendConfig, SdpaParameters
from ..protocol import VarlenSdpaBackend


class FlashAttention2VarlenSdpa(nn.Module, VarlenSdpaBackend):
    """Variable-length Scaled Dot Product Attention using Flash Attention 2.

    Args:
        config: Backend configuration.
        params: Structural parameters.
    """

    def __init__(self, config: FlashAttention2SdpaBackendConfig, params: SdpaParameters) -> None:
        super().__init__()

        if params.num_sinks is not None:
            raise ValueError("Flash Attention 2 varlen backend does not support learnable sinks (`num_sinks`).")

        if params.needs_attention_mask:
            raise ValueError("Flash Attention 2 varlen backend does not support explicit attention masks.")

        self._window_size = params.window_size

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        left = self._window_size[0] if self._window_size[0] is not None else -1
        right = self._window_size[1] if self._window_size[1] is not None else -1
        window = (left, right)

        return flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=is_causal,
            window_size=window,
        )
