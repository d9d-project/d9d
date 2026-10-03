import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention import GroupedQueryAttention
from d9d.module.block.ffn import SwiGLU
from d9d.module.block.normalization import RMSNorm
from d9d.module.block.positional import RotaryEmbeddingStyle

from .params import Qwen3DenseLayerParameters


class Qwen3DenseLayer(nn.Module, ModuleLateInit):
    """A single Qwen3 Dense transformer layer.

    The layer applies Grouped Query Attention, then a SwiGLU MLP block. Each sub-layer has a
    pre-RMSNorm and a residual connection.
    """

    def __init__(self, params: Qwen3DenseLayerParameters):
        """Constructs the ``Qwen3DenseLayer`` object.

        Args:
            params: Configuration parameters for the layer.
        """
        super().__init__()

        self.self_attn = GroupedQueryAttention(
            hidden_size=params.hidden_size,
            num_attention_heads=params.num_attention_heads,
            num_key_value_heads=params.num_key_value_heads,
            is_causal=True,
            qk_norm_eps=params.rms_norm_eps,
            head_dim=params.head_dim,
            rope_style=RotaryEmbeddingStyle.HALF,
        )

        self.mlp = SwiGLU(hidden_size=params.hidden_size, intermediate_size=params.intermediate_size, bias=False)

        self.input_layernorm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Runs the forward pass of the layer.

        Args:
            hidden_states: Input hidden states. Shape: ``(batch, seq_len, hidden_size)``.
            position_embeddings: Precomputed RoPE embeddings as a ``(cos, sin)`` tuple.

        Returns:
            Output hidden states. Shape: ``(batch, seq_len, hidden_size)``.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

    def reset_parameters(self):
        """Resets the module parameters."""
        self.self_attn.reset_parameters()
        self.mlp.reset_parameters()
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
