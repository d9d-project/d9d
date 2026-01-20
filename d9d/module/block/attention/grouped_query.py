import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention.sdpa import FlashSdpa
from d9d.module.block.positional import RotaryEmbeddingApplicator


class GroupedQueryAttention(nn.Module, ModuleLateInit):
    """
    Implements Grouped Query Attention (GQA) with RoPE and optional QK Normalization.

    This module performs the full attention mechanism pipeline:
    1.  Linear projection to Q, K, V.
    2.  Optional RMS Normalization on Q and K.
    3.  Rotary Positional Embedding (RoPE) application.
    4.  Scaled Dot Product Attention (via FlashAttention).
    5.  Output projection.
    """

    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            num_key_value_heads: int,
            head_dim: int,
            qk_norm_eps: float | None,
            is_causal: bool
    ):
        """
        Constructs the GroupedQueryAttention layer.

        Args:
            hidden_size: Hidden size.
            num_attention_heads: Number of Query heads.
            num_key_value_heads: Number of Key/Value heads. If less than `num_attention_heads`, GQA/MQA is enabled.
            head_dim: Dimensionality of a single attention head.
            qk_norm_eps: Epsilon for LayerNorm/RMSNorm applied to Q and K. If None, normalization is disabled.
            is_causal: Whether to apply a causal mask (auto-regressive constraint).
        """

        super().__init__()

        self._head_dim = head_dim
        self._num_key_value_groups = num_attention_heads // num_key_value_heads
        self._scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )

        self.q_norm: nn.RMSNorm | None
        self.k_norm: nn.RMSNorm | None

        if qk_norm_eps is not None:
            self.q_norm = nn.RMSNorm(normalized_shape=head_dim,
                                     eps=qk_norm_eps)
            self.k_norm = nn.RMSNorm(normalized_shape=head_dim,
                                     eps=qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rope = RotaryEmbeddingApplicator()
        self.kernel = FlashSdpa()
        self._is_causal = is_causal

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the attention operation.

        Args:
            hidden_states: Input tensor. Shape: `(batch, seq_len, hidden_size)`.
            attention_mask: Optional mask associated with the inputs.
            position_embeddings: Tuple of `(cos, sin)` tensors for RoPE application.
                Each tensor should be of shape `(batch, seq_len, head_dim)`

        Returns:
            The attention output tensor. Shape: `(batch, seq_len, hidden_size)`.
        """

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self._head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = self.rope(query_states, key_states, position_embeddings[0], position_embeddings[1])

        outputs = self.kernel(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            is_causal=self._is_causal,
            scale=self._scaling
        )

        outputs = outputs.reshape(*input_shape, -1).contiguous()
        outputs = self.o_proj(outputs)
        return outputs

    def reset_parameters(self):
        """Resets module parameters."""

        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.o_proj.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
