import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class FlashSdpa(nn.Module):
    """Executes Scaled Dot Product Attention (SDPA) enforcing the FlashAttention backend."""

    def __init__(self):
        """
        Constructs the FlashSdpa object.
        """
        super().__init__()

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        """
        Computes Scaled Dot-Product Attention using FlashAttention.

        Args:
            query_states: Query tensor. Shape: `(batch, n_q_heads, seq_len, head_dim)`.
            key_states: Key tensor. Shape: `(batch, n_kv_heads, seq_len, head_dim)`.
            value_states: Value tensor. Shape: `(batch, n_kv_heads, seq_len, head_dim)`.
            attention_mask: Optional attention mask (usually not needed for FlashAttn with causal=True).
            is_causal: If True, applies a causal mask (upper triangular masking).
            scale: Scaling factor applied to the dot products (usually `1 / sqrt(head_dim)`).

        Returns:
            The attention output tensor, permuted to channel-last format.
                Shape: `(batch, seq_len, n_q_heads, head_dim)`.
        """

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            results = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=query_states.shape[1] != key_states.shape[1],
            )
            return results.transpose(1, 2).contiguous()
