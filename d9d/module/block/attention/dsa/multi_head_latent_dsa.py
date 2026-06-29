import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention.dsa.lightning_indexer import LightningIndexer, build_sparse_selection_mask
from d9d.module.block.attention.multi_head_latent import MultiHeadLatentAttention
from d9d.module.block.attention.sdpa import AnySdpaBackendConfig, TorchSdpaBackendConfig
from d9d.module.block.positional import RotaryEmbeddingStyle


class MultiHeadLatentSparseAttention(nn.Module, ModuleLateInit):
    """Implements DeepSeek Sparse Attention (DSA) instantiated under MLA.

    This is the instantiation used by DeepSeek-V3.2. Because MLA shares a
    single latent key-value entry across all query heads (its MQA mode), the top-k
    selection picks ``k`` latent vectors per query, which is what makes the sparse gather
    cheap for inference; in this dense-mask training formulation the selection is implemented
    as an additive mask.

    A lightweight ``LightningIndexer`` scores every preceding token for each query; only
    the top-k highest scoring tokens are kept, all other positions are masked out before
    the softmax, and causality is folded into the same additive mask (so MLA runs with
    ``is_causal=False`` and any mask-capable SDPA backend can be used).

    References:
        [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: int | None,
        qk_down_norm_eps: float,
        index_n_heads: int,
        index_head_dim: int,
        index_top_k: int,
        rope_style: RotaryEmbeddingStyle,
        sdpa_backend: AnySdpaBackendConfig | None = None,
    ) -> None:
        """Constructs the MultiHeadLatentSparseAttention layer.

        Args:
            hidden_size: Model hidden dimension.
            num_attention_heads: Number of attention heads.
            qk_nope_head_dim: Per-head dimension for the content (no-RoPE) part of Q and K.
            qk_rope_head_dim: Per-head dimension for the RoPE-rotated part of Q and K.
            v_head_dim: Per-head dimension for values.
            kv_lora_rank: Rank of the KV latent compression.
            q_lora_rank: Rank of the Q low-rank path. If ``None``, Q is projected directly.
            qk_down_norm_eps: Epsilon for the RMSNorm applied to the KV and Q latent representations.
            index_n_heads: Number of lightning-indexer heads (``H_I``).
            index_head_dim: Per-head dimension of the lightning indexer (``d_I``).
            index_top_k: Number of tokens each query attends to after selection (``k``).
            rope_style: Rotary embedding layout style alignment.
            sdpa_backend: Configuration for the Scaled Dot-Product Attention backend. The backend must
                accept an explicit attention mask; if ``None``, the PyTorch SDPA backend is used.
        """
        super().__init__()

        # Causality is encoded in the additive selection mask, so the inner attention is non-causal
        # and only needs to support an explicit mask.
        self.attention = MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_down_norm_eps=qk_down_norm_eps,
            is_causal=False,
            rope_style=rope_style,
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
        """Computes sparse latent attention with lightning-indexer token selection.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional additive mask (e.g. for padding) broadcastable to
                ``(batch, num_heads, seq_len, seq_len)``. Added on top of the selection and causal masks.
            position_embeddings: Tuple of ``(cos, sin)`` tensors for the RoPE sub-vectors.

        Returns:
            The attention output tensor. Shape: ``(batch, seq_len, hidden_size)``.
        """
        sparse_mask = build_sparse_selection_mask(self.indexer, hidden_states, attention_mask)
        return self.attention(
            hidden_states=hidden_states,
            attention_mask=sparse_mask,
            position_embeddings=position_embeddings,
        )

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.attention.reset_parameters()
        self.indexer.reset_parameters()
