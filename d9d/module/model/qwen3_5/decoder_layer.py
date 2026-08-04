import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention import GroupedQueryAttention
from d9d.module.block.attention.linear import GatedDeltaNet, MambaDecayGateParameters
from d9d.module.block.ffn import SwiGLU
from d9d.module.block.normalization import RMSNorm
from d9d.module.block.positional import RotaryEmbeddingStyle

from .params import Qwen3p5LayerParameters

# Mamba-style decay gate constants used by the Qwen3-Next/Qwen3.5 family.
_DECAY_GATE_NORMALIZER = 16.0
_DECAY_GATE_DT_MIN = 0.001
_DECAY_GATE_DT_MAX = 0.1
_DECAY_GATE_DT_INIT_FLOOR = 1e-4


class Qwen3p5FullAttentionLayer(nn.Module, ModuleLateInit):
    """Implements a Qwen3.5 dense decoder layer with full (softmax) attention.

    This layer consists of a Grouped Query Attention mechanism with partial RoPE, sigmoid output
    gating and zero-centered QK normalization, followed by a SwiGLU MLP block, with pre-RMSNorm
    (zero-centered) applied before each sub-layer.
    """

    def __init__(self, params: Qwen3p5LayerParameters):
        """Constructs a Qwen3p5FullAttentionLayer object.

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
            rope_dim=params.rope_dim,
            enable_output_gate=True,
            qk_norm_zero_centered=True,
        )

        self.mlp = SwiGLU(
            hidden_size=params.hidden_size,
            intermediate_size=params.intermediate_size,
            bias=False,
        )

        self.input_layernorm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps, zero_centered=True)
        self.post_attention_layernorm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps, zero_centered=True)

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Performs the forward pass of the full-attention layer.

        Args:
            hidden_states: Input tensor of shape `(batch, seq_len, hidden_dim)`.
            position_embeddings: Tuple containing RoPE precomputed embeddings (cos, sin).

        Returns:
            Output tensor after attention and MLP blocks, shape `(batch, seq_len, hidden_dim)`.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,  # no mask for the decoder
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def reset_parameters(self):
        """Resets module parameters."""
        self.self_attn.reset_parameters()
        self.mlp.reset_parameters()
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()


class Qwen3p5LinearAttentionLayer(nn.Module, ModuleLateInit):
    """Implements a Qwen3.5 dense decoder layer with linear attention (Gated DeltaNet).

    This layer consists of a Gated DeltaNet token mixer followed by a SwiGLU MLP block, with
    pre-RMSNorm (zero-centered) applied before each sub-layer.
    """

    def __init__(self, params: Qwen3p5LayerParameters):
        """Constructs a Qwen3p5LinearAttentionLayer object.

        Args:
            params: Configuration parameters for the layer.
        """
        super().__init__()

        self.linear_attn = GatedDeltaNet(
            hidden_size=params.hidden_size,
            num_query_key_heads=params.linear_num_key_heads,
            num_value_heads=params.linear_num_value_heads,
            head_qk_dim=params.linear_key_head_dim,
            head_v_dim=params.linear_value_head_dim,
            norm_eps=params.rms_norm_eps,
            conv_size=params.linear_conv_kernel_dim,
            decay_gate=MambaDecayGateParameters(
                normalizer=_DECAY_GATE_NORMALIZER,
                dt_min=_DECAY_GATE_DT_MIN,
                dt_max=_DECAY_GATE_DT_MAX,
                dt_init_floor=_DECAY_GATE_DT_INIT_FLOOR,
            ),
        )

        self.mlp = SwiGLU(
            hidden_size=params.hidden_size,
            intermediate_size=params.intermediate_size,
            bias=False,
        )

        self.input_layernorm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps, zero_centered=True)
        self.post_attention_layernorm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps, zero_centered=True)

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Performs the forward pass of the linear-attention layer.

        Args:
            hidden_states: Input tensor of shape `(batch, seq_len, hidden_dim)`.
            position_embeddings: Tuple containing RoPE precomputed embeddings (cos, sin).
                Unused by the linear attention mixer; accepted for interface uniformity with the
                full-attention layer.

        Returns:
            Output tensor after linear attention and MLP blocks, shape
            `(batch, seq_len, hidden_dim)`.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def reset_parameters(self):
        """Resets module parameters."""
        self.linear_attn.reset_parameters()
        self.mlp.reset_parameters()
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
