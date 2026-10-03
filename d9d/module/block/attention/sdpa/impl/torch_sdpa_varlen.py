import contextlib

import torch
import torch.nn.functional as F
from torch import nn

from ..config import SdpaParameters, TorchSdpaBackendConfig
from ..protocol import VarlenSdpaBackend
from .torch_sdpa import _backend_type_to_torch


def _build_block_diagonal_mask(cu_seqlens: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Builds a block-diagonal boolean attention mask from cumulative segment lengths.

    Args:
        cu_seqlens: Cumulative segment lengths, shape ``(num_segments + 1,)``.
        total_tokens: The total number of packed tokens.

    Returns:
        A boolean mask of shape ``(total_tokens, total_tokens)`` where ``True`` marks
        attendable positions (same segment).
    """
    segment_ids = torch.zeros(total_tokens, dtype=torch.long, device=cu_seqlens.device)
    segment_ids[cu_seqlens[1:-1].long()] = 1
    segment_ids = segment_ids.cumsum(dim=0)
    return segment_ids[:, None] == segment_ids[None, :]


class TorchVarlenSdpa(nn.Module, VarlenSdpaBackend):
    """Variable-length Scaled Dot Product Attention using PyTorch's `scaled_dot_product_attention`.

    Materializes a block-diagonal attention mask from the cumulative segment lengths. Correct but
    quadratic in the total packed length; intended as a dependency-free fallback.

    Args:
        config: Backend configuration.
        params: Structural parameters.
    """

    def __init__(self, config: TorchSdpaBackendConfig, params: SdpaParameters) -> None:
        super().__init__()

        if params.num_sinks is not None:
            raise ValueError("PyTorch varlen SDPA backend does not support learnable sinks (`num_sinks`).")

        if params.window_size != (None, None):
            raise ValueError("PyTorch varlen SDPA backend does not support sliding window attention.")

        if config.backends is not None:
            self._backends = [_backend_type_to_torch(backend) for backend in config.backends]
        else:
            self._backends = None

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
        total_tokens = query_states.shape[0]

        mask = _build_block_diagonal_mask(cu_seqlens, total_tokens)
        if is_causal:
            causal = torch.tril(torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=mask.device))
            mask = mask & causal

        # (total, heads, dim) -> (1, heads, total, dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        ctx = torch.nn.attention.sdpa_kernel(self._backends) if self._backends is not None else contextlib.nullcontext()
        is_gqa = query_states.shape[1] != key_states.shape[1]

        with ctx:
            out = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=mask,
                scale=scale,
                enable_gqa=is_gqa,
            )

        return out.squeeze(0).transpose(0, 1).contiguous()
