from typing import Protocol

import torch


class SdpaBackend(Protocol):
    """Protocol for Scaled Dot-Product Attention backends.

    This acts as a structural trait for attention backend modules. Any backend
    must provide a `__call__` method with this signature since PyTorch modules
    are invoked directly.
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
        """Computes Scaled Dot-Product Attention.

        Args:
            query_states: Query tensor. Shape: ``(batch, seq_len, n_q_heads, head_dim)``.
            key_states: Key tensor. Shape: ``(batch, seq_len, n_kv_heads, head_dim)``.
            value_states: Value tensor. Shape: ``(batch, seq_len, n_kv_heads, head_dim)``.
            attention_mask: Mask tensor or None.
            is_causal: If True, applies an auto-regressive causal mask.
            scale: Softmax scaling factor.

        Returns:
            Attention output. Shape: ``(batch, seq_len, n_q_heads, head_dim)``.
        """
        ...
