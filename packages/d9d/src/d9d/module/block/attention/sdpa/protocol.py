from typing import Protocol

import torch


class SdpaBackend(Protocol):
    """Protocol for scaled dot-product attention (SDPA) backends.

    Backends are ``nn.Module`` objects that are called directly, so the protocol declares ``__call__``.
    """

    def __call__(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        """Computes scaled dot-product attention.

        Args:
            query_states: Query tensor. Shape: ``(batch, seq_len, num_heads, head_dim)``.
            key_states: Key tensor. Shape: ``(batch, seq_len, num_kv_heads, head_dim)``.
            value_states: Value tensor. Shape: ``(batch, seq_len, num_kv_heads, head_dim)``.
            attention_mask: Optional attention mask, or ``None``.
            is_causal: If ``True``, applies a causal mask.
            scale: Softmax scaling factor.

        Returns:
            Attention output. Shape: ``(batch, seq_len, num_heads, head_dim)``.
        """
        ...
