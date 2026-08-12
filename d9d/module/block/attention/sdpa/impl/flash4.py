import torch
import torch.nn.functional as F
from torch import nn

from d9d.kernel.flash_attn import flash_attn_func, flash_attn_varlen_func

from ...types import SequencePacking
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

    @staticmethod
    def _pad_head_dim(*tensors: torch.Tensor) -> tuple[int, tuple[torch.Tensor, ...]]:
        """Zero-pads the head_dim of each tensor up to a multiple of ``_FA4_HDIM_ALIGN``.

        FA4's backward preprocess kernel requires an aligned head_dim; the zero padding is transparent
        (extra dims contribute 0 to QK^T and to the weighted sum over V) and is stripped from the
        output afterwards.

        Returns:
            The amount padded (0 when already aligned) and the padded tensors.
        """
        head_dim = tensors[0].shape[-1]
        pad = (-head_dim) % _FA4_HDIM_ALIGN
        if pad:
            tensors = tuple(F.pad(tensor, (0, pad)) for tensor in tensors)
        return pad, tensors

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
            raise ValueError("Flash Attention 4 does not support setting attention mask explicitly")

        if packing is not None and query_states.shape[0] != 1:
            raise ValueError(
                "Flash Attention 4 sequence packing expects a single packed row (batch dimension "
                f"of size one), got batch size {query_states.shape[0]}"
            )

        head_dim = query_states.shape[-1]
        pad, (query_states, key_states, value_states) = self._pad_head_dim(query_states, key_states, value_states)

        if packing is None:
            out, *_ = flash_attn_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=scale,
                causal=is_causal,
                window_size=self._window_size,
                learnable_sink=self.sinks,
            )
        else:
            # (1, total, heads, dim) -> (total, heads, dim) for the varlen kernel.
            out, *_ = flash_attn_varlen_func(
                query_states.squeeze(0),
                key_states.squeeze(0),
                value_states.squeeze(0),
                cu_seqlens_q=packing.cu_seqlens,
                cu_seqlens_k=packing.cu_seqlens,
                max_seqlen_q=packing.max_seqlen,
                max_seqlen_k=packing.max_seqlen,
                softmax_scale=scale,
                causal=is_causal,
                window_size=self._window_size,
                learnable_sink=self.sinks,
            )
            # (total, heads, dim) -> (1, total, heads, dim).
            out = out.unsqueeze(0)
        if pad:
            out = out[..., :head_dim]
        return out
