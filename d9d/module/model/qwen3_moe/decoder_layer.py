import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention import GroupedQueryAttention
from d9d.module.block.moe import MoELayer
from d9d.module.model.qwen3_moe import Qwen3MoELayerParameters


class Qwen3MoELayer(nn.Module, ModuleLateInit):
    """
    Implements a single Qwen3 Mixture-of-Experts (MoE) transformer layer.

    This layer consists of a Grouped Query Attention mechanism followed by an MoE
    MLP block, with pre-RMSNorm applied before each sub-layer.
    """

    def __init__(
            self,
            params: Qwen3MoELayerParameters
    ):
        """
        Constructs a Qwen3MoELayer object.

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
            head_dim=params.head_dim
        )

        self.mlp = MoELayer(
            hidden_dim=params.hidden_size,
            num_grouped_experts=params.num_experts,
            intermediate_dim_grouped=params.intermediate_size,
            top_k=params.experts_top_k,
            router_renormalize_probabilities=True
        )

        self.input_layernorm = nn.RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(params.hidden_size, eps=params.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs the forward pass of the MoE layer.

        Args:
            hidden_states: Input tensor of shape `(batch, seq_len, hidden_dim)`.
            position_embeddings: Tuple containing RoPE precomputed embeddings (cos, sin).

        Returns:
            Output tensor after attention and MoE blocks, shape `(batch, seq_len, hidden_dim)`.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None  # no mask for moe decoder
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

    def reset_moe_stats(self):
        """
        Resets statistical counters inside the MoE router (e.g., token counts per expert).
        """

        self.mlp.reset_stats()

    @property
    def moe_tokens_per_expert(self) -> torch.Tensor:
        """
        Returns the number of tokens routed to each expert.
        """

        return self.mlp.tokens_per_expert

    def reset_parameters(self):
        """
        Resets module parameters.
        """

        self.self_attn.reset_parameters()
        self.mlp.reset_parameters()
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
