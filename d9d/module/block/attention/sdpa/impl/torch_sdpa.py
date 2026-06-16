import contextlib

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend

from ..config import SdpaParameters, TorchSdpaBackendConfig, TorchSdpaBackendType
from ..protocol import SdpaBackend


def _backend_type_to_torch(backend: TorchSdpaBackendType) -> SDPBackend:
    match backend:
        case TorchSdpaBackendType.MATH:
            return SDPBackend.MATH
        case TorchSdpaBackendType.FLASH_ATTENTION:
            return SDPBackend.FLASH_ATTENTION
        case TorchSdpaBackendType.EFFICIENT_ATTENTION:
            return SDPBackend.EFFICIENT_ATTENTION
        case TorchSdpaBackendType.CUDNN_ATTENTION:
            return SDPBackend.CUDNN_ATTENTION


class TorchSdpa(nn.Module, SdpaBackend):
    """Scaled Dot Product Attention using PyTorch's eager `scaled_dot_product_attention`.

    Args:
        config: Backend configuration.
        params: Structural parameters.
    """

    def __init__(self, config: TorchSdpaBackendConfig, params: SdpaParameters) -> None:
        super().__init__()

        if params.num_sinks is not None:
            raise ValueError("PyTorch SDPA backend does not support learnable sinks (`num_sinks`).")

        if params.window_size != (None, None):
            raise ValueError("PyTorch SDPA backend does not support sliding window attention.")

        if config.backends is not None:
            self._backends = [_backend_type_to_torch(backend) for backend in config.backends]
        else:
            self._backends = None

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        ctx = torch.nn.attention.sdpa_kernel(self._backends) if self._backends is not None else contextlib.nullcontext()
        is_gqa = query_states.shape[1] != key_states.shape[1]

        with ctx:
            out = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=is_gqa,
                attn_mask=attention_mask,
            )

        out = out.transpose(1, 2).contiguous()

        return out
