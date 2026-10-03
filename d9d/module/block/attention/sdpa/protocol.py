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


class VarlenSdpaBackend(Protocol):
    """Protocol for variable-length Scaled Dot-Product Attention backends.

    Operates on packed sequences: multiple segments concatenated along the token dimension, with
    segment boundaries given by cumulative sequence lengths. Attention never crosses segment
    boundaries.

    This acts as a structural trait for attention backend modules. Any backend must provide a
    `__call__` method with this signature since PyTorch modules are invoked directly.
    """

    def __call__(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        """Computes Scaled Dot-Product Attention over packed segments.

        Args:
            query_states: Query tensor. Shape: ``(total_tokens, n_q_heads, head_dim)``.
            key_states: Key tensor. Shape: ``(total_tokens, n_kv_heads, head_dim)``.
            value_states: Value tensor. Shape: ``(total_tokens, n_kv_heads, head_dim)``.
            cu_seqlens: Cumulative segment lengths, shape ``(num_segments + 1,)``, dtype int32.
            max_seqlen: The length of the longest segment.
            is_causal: If True, applies an auto-regressive causal mask within each segment.
            scale: Softmax scaling factor.

        Returns:
            Attention output. Shape: ``(total_tokens, n_q_heads, head_dim)``.
        """
        ...
