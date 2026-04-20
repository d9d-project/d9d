import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention.sdpa import FlashSdpa
from d9d.module.block.normalization import RMSNorm
from d9d.module.block.positional import RotaryEmbeddingApplicator, RotaryEmbeddingStyle


class GroupedQueryAttention(nn.Module, ModuleLateInit):
    """
    Implements Grouped Query Attention (GQA) with RoPE and optional QK Normalization.

    This module performs the full attention mechanism pipeline:
    1.  Linear projection to Q, K, V.
    2.  Optional RMS Normalization on Q and K.
    3.  Rotary Positional Embedding (RoPE) application.
    4.  Scaled Dot Product Attention (via FlashAttention).
    5.  Optional sigmoid output gating.
    6.  Output projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        qk_norm_eps: float | None,
        is_causal: bool,
        rope_style: RotaryEmbeddingStyle,
        rope_dim: int | None = None,
        enable_output_gate: bool = False,
        qk_norm_zero_centered: bool = False,
    ) -> None:
        """
        Constructs the GroupedQueryAttention layer.

        Args:
            hidden_size: Hidden size.
            num_attention_heads: Number of Query heads.
            num_key_value_heads: Number of Key/Value heads. If less than `num_attention_heads`, GQA/MQA is enabled.
            head_dim: Dimensionality of a single attention head.
            qk_norm_eps: Epsilon for LayerNorm/RMSNorm applied to Q and K. If None, normalization is disabled.
            is_causal: Whether to apply a causal mask (auto-regressive constraint).
            rope_style: Rotary embedding layout style alignment.
            rope_dim: Dimension of the RoPE sub-vector. If ``None``, RoPE is applied to the full ``head_dim``.
            enable_output_gate: If True, enables sigmoid output gating (Qwen 3.5 style).
            qk_norm_zero_centered: If True, utilizes zero-centered scaling weights for the optional Q and K
                normalization layers.
        """

        super().__init__()

        self._head_dim = head_dim
        self._num_key_value_groups = num_attention_heads // num_key_value_heads
        self._scaling = head_dim**-0.5
        self._rope_dim: int | None = rope_dim
        self._nope_dim: int = head_dim - rope_dim if rope_dim is not None else 0

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        if enable_output_gate:
            self.gate_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        else:
            self.gate_proj = None

        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)

        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)

        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.q_norm: RMSNorm | None
        self.k_norm: RMSNorm | None

        if qk_norm_eps is not None:
            self.q_norm = RMSNorm(head_dim, eps=qk_norm_eps, zero_centered=qk_norm_zero_centered)
            self.k_norm = RMSNorm(head_dim, eps=qk_norm_eps, zero_centered=qk_norm_zero_centered)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rope = RotaryEmbeddingApplicator(style=rope_style)
        self.kernel = FlashSdpa()
        self._is_causal = is_causal

    def _apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope_dim is not None:
            q_rope, q_nope = query_states.split([self._rope_dim, self._nope_dim], dim=-1)
            k_rope, k_nope = key_states.split([self._rope_dim, self._nope_dim], dim=-1)
            q_rope, k_rope = self.rope(q_rope, k_rope, cos, sin)
            return torch.cat([q_rope, q_nope], dim=-1), torch.cat([k_rope, k_nope], dim=-1)
        return self.rope(query_states, key_states, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the attention operation.

        Args:
            hidden_states: Input tensor. Shape: `(batch, seq_len, hidden_size)`.
            attention_mask: Optional mask associated with the inputs.
            position_embeddings: Tuple of `(cos, sin)` tensors for RoPE application.
                Each tensor should be of shape `(batch, seq_len, rope_dim)` when partial RoPE is used,
                or `(batch, seq_len, head_dim)` otherwise.

        Returns:
            The attention output tensor. Shape: `(batch, seq_len, hidden_size)`.
        """

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self._head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rope(query_states, key_states, cos, sin)

        outputs = self.kernel(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            is_causal=self._is_causal,
            scale=self._scaling,
        )

        outputs = outputs.reshape(*input_shape, -1)

        if self.gate_proj is not None:
            outputs = outputs * torch.sigmoid(self.gate_proj(hidden_states))

        outputs = self.o_proj(outputs)
        return outputs

    def reset_parameters(self) -> None:
        """Resets module parameters."""

        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()

        if self.gate_proj is not None:
            self.gate_proj.reset_parameters()

        self.o_proj.reset_parameters()

        if self.q_norm is not None:
            self.q_norm.reset_parameters()

        if self.k_norm is not None:
            self.k_norm.reset_parameters()
