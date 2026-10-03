import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention.sdpa import (
    AnySdpaBackendConfig,
    SdpaParameters,
    VarlenSdpaBackend,
    build_varlen_sdpa_backend,
)
from d9d.module.block.positional import RotaryEmbeddingApplicator, RotaryEmbeddingStyle


class PackedVisionAttention(nn.Module, ModuleLateInit):
    """Bidirectional multi-head attention over packed variable-length media segments.

    Operates on a packed token stream ``(total_tokens, hidden)``; attention never crosses segment
    boundaries, which are provided via cumulative sequence lengths.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        sdpa_backend: AnySdpaBackendConfig | None = None,
    ):
        """Constructs a PackedVisionAttention object.

        Args:
            hidden_size: The vision encoder hidden size.
            num_attention_heads: Number of attention heads.
            sdpa_backend: Explicit varlen SDPA backend configuration, or ``None`` to auto-detect.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_attention_heads``.
        """
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})."
            )

        self._num_heads = num_attention_heads
        self._head_dim = hidden_size // num_attention_heads
        self._scale = self._head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.rope = RotaryEmbeddingApplicator(style=RotaryEmbeddingStyle.HALF)
        self.kernel: VarlenSdpaBackend = build_varlen_sdpa_backend(
            params=SdpaParameters(num_sinks=None, window_size=(None, None), needs_attention_mask=False),
            backend_config=sdpa_backend,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the attention operation.

        Args:
            hidden_states: Packed input tensor. Shape: ``(total_tokens, hidden_size)``.
            cu_seqlens: Cumulative segment lengths, shape ``(num_segments + 1,)``, dtype int32.
            max_seqlen: The length of the longest segment.
            position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application, each of
                shape ``(total_tokens, head_dim)``.

        Returns:
            The attention output tensor. Shape: ``(total_tokens, hidden_size)``.
        """
        total_tokens = hidden_states.shape[0]

        query_states, key_states, value_states = (
            self.qkv(hidden_states).view(total_tokens, 3, self._num_heads, self._head_dim).unbind(dim=1)
        )

        cos, sin = position_embeddings
        query_states, key_states = self.rope(
            query_states.unsqueeze(0),
            key_states.unsqueeze(0),
            cos.unsqueeze(0),
            sin.unsqueeze(0),
        )
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)

        outputs = self.kernel(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            is_causal=False,
            scale=self._scale,
        )

        return self.proj(outputs.reshape(total_tokens, -1))

    def reset_parameters(self):
        """Resets module parameters."""
        self.qkv.reset_parameters()
        self.proj.reset_parameters()
