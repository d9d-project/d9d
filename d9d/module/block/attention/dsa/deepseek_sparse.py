import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention.dsa.lightning_indexer import LightningIndexer
from d9d.module.block.attention.grouped_query import GroupedQueryAttention
from d9d.module.block.attention.sdpa import AnySdpaBackendConfig, TorchSdpaBackendConfig
from d9d.module.block.positional import RotaryEmbeddingStyle


def _build_causal_bias(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Builds an additive causal bias of shape ``(seq_len, seq_len)``.

    Entry ``(q, k)`` is ``0`` when key ``k`` may be attended by query ``q`` (``k <= q``)
    and ``-inf`` otherwise.

    Returns:
        The additive causal bias tensor.
    """
    positions = torch.arange(seq_len, device=device)
    disallowed = positions.unsqueeze(0) > positions.unsqueeze(1)
    return torch.zeros(seq_len, seq_len, device=device, dtype=dtype).masked_fill_(disallowed, float("-inf"))


class DeepSeekSparseAttention(nn.Module, ModuleLateInit):
    """Implements DeepSeek Sparse Attention (DSA).

    DSA augments ordinary causal attention with a fine-grained token selection
    mechanism. A lightweight ``LightningIndexer`` scores every preceding token for
    each query; only the top-k highest scoring tokens are attended to, while all
    other positions are masked out before the softmax. This preserves the quality
    of dense attention in long-context settings while reducing the core attention
    to attend over k (<< L) tokens per query.

    The selection is realised as an additive mask, so the actual attention is
    delegated to a standard ``GroupedQueryAttention`` backend (which also handles
    the Q/K/V projections, optional QK-Norm, RoPE and output gating). Causality is
    folded into the additive mask, hence the inner attention runs with
    ``is_causal=False`` and any mask-capable SDPA backend can be used.

    References:
        [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_top_k: int,
        qk_norm_eps: float | None,
        rope_style: RotaryEmbeddingStyle,
        rope_dim: int | None = None,
        enable_output_gate: bool = False,
        qk_norm_zero_centered: bool = False,
        sdpa_backend: AnySdpaBackendConfig | None = None,
    ) -> None:
        """Constructs the DeepSeekSparseAttention layer.

        Args:
            hidden_size: Hidden size.
            num_attention_heads: Number of Query heads.
            num_key_value_heads: Number of Key/Value heads. If less than `num_attention_heads`, GQA/MQA is enabled.
            head_dim: Dimensionality of a single attention head.
            index_n_heads: Number of lightning-indexer heads (``H_I``).
            index_head_dim: Per-head dimension of the lightning indexer (``d_I``).
            index_top_k: Number of tokens each query attends to after selection (``k``).
            qk_norm_eps: Epsilon for the RMSNorm applied to Q and K. If None, normalization is disabled.
            rope_style: Rotary embedding layout style alignment.
            rope_dim: Dimension of the RoPE sub-vector. If ``None``, RoPE is applied to the full ``head_dim``.
            enable_output_gate: If True, enables sigmoid output gating (Qwen 3.5 style).
            qk_norm_zero_centered: If True, utilizes zero-centered scaling weights for the optional Q and K
                RMSNorm layers (DeepSeek V3 style).
            sdpa_backend: Configuration for the Scaled Dot-Product Attention backend. The backend must
                accept an explicit attention mask; if ``None``, the PyTorch SDPA backend is used.
        """
        super().__init__()

        # Causality is encoded in the additive selection mask, so the inner attention is non-causal
        # and only needs to support an explicit mask.
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            qk_norm_eps=qk_norm_eps,
            is_causal=False,
            rope_style=rope_style,
            rope_dim=rope_dim,
            enable_output_gate=enable_output_gate,
            qk_norm_zero_centered=qk_norm_zero_centered,
            sdpa_backend=sdpa_backend if sdpa_backend is not None else TorchSdpaBackendConfig(),
        )
        self.indexer = LightningIndexer(
            hidden_size=hidden_size,
            num_heads=index_n_heads,
            head_dim=index_head_dim,
            top_k=index_top_k,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes sparse attention with lightning-indexer token selection.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional additive mask (e.g. for padding) broadcastable to
                ``(batch, num_heads, seq_len, seq_len)``. Added on top of the selection and causal masks.
            position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application.

        Returns:
            The attention output tensor. Shape: ``(batch, seq_len, hidden_size)``.
        """
        seq_len = hidden_states.shape[1]
        causal_bias = _build_causal_bias(seq_len, hidden_states.device, hidden_states.dtype)

        # Rank tokens with the causal bias so future positions are never preferred, then enforce
        # causality in the final mask as well (top-k may still include future slots when k > t).
        selection_mask = self.indexer(hidden_states, attention_bias=causal_bias)
        sparse_mask = (selection_mask + causal_bias).unsqueeze(1)

        if attention_mask is not None:
            sparse_mask = sparse_mask + attention_mask

        return self.attention(
            hidden_states=hidden_states,
            attention_mask=sparse_mask,
            position_embeddings=position_embeddings,
        )

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.attention.reset_parameters()
        self.indexer.reset_parameters()
