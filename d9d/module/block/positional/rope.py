import torch
from torch import nn

from d9d.module.base import ModuleLateInit


def _prepare_rope_inverse_frequencies(rope_base: int, inside_dim: int) -> torch.Tensor:
    """
    Calculates inverse frequencies for RoPE calculation.

    Args:
        rope_base: Base for the geometric progression.
        inside_dim: Dimension of the attention head (must be even).

    Returns:
        A tensor containing the inverse frequencies.
    """

    power = torch.arange(0, inside_dim, 2, dtype=torch.int64).to(dtype=torch.float) / inside_dim
    freq = rope_base**power
    inv_freq = 1.0 / freq
    return inv_freq


def prepare_rotary_cos_sin_emb(
    rope_base: int, head_dim: int, max_position_ids: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes rotary cosine and sine embeddings.

    Args:
        rope_base: Base frequency for calculation.
        head_dim: Dimensionality of the attention head (E).
        max_position_ids: Maximum sequence length supported (S).
        device: Target device for the tensors.
        dtype: Target data type for the tensors.

    Returns:
        A tuple containing cosine and sine tensors, both of shapes [S, E].
    """

    position_ids = torch.arange(0, max_position_ids, dtype=torch.long)
    freqs = _prepare_rope_inverse_frequencies(rope_base, head_dim)

    arguments = (freqs[:, None] @ position_ids[None, :].float()).T

    emb = torch.cat((arguments, arguments), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)


class RotaryEmbeddingProvider(nn.Module, ModuleLateInit):
    """Module that manages and provides Rotary Positional Embeddings."""

    def __init__(self, rope_base: int, head_dim: int, max_position_ids: int):
        """Constructs the RotaryEmbeddingProvider."""

        super().__init__()
        self._rope_base = rope_base
        self._head_dim = head_dim
        self._max_position_ids = max_position_ids
        self.cos_emb = nn.Buffer(torch.empty(max_position_ids, head_dim), persistent=False)
        self.sin_emb = nn.Buffer(torch.empty(max_position_ids, head_dim), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves cached cosine and sine embeddings for specific positions.

        Args:
            position_ids: Tensor of position indices.

        Returns:
            A tuple of (cos, sin) tensors aligned with the input positions.
        """

        return self.cos_emb[position_ids], self.sin_emb[position_ids]

    def reset_parameters(self):
        with torch.no_grad():
            cos, sin = prepare_rotary_cos_sin_emb(
                rope_base=self._rope_base,
                head_dim=self._head_dim,
                max_position_ids=self._max_position_ids,
                device=self.cos_emb.device,
                dtype=self.cos_emb.dtype,
            )
            self.cos_emb.data = cos
            self.sin_emb.data = sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbeddingApplicator(nn.Module):
    """Applies Rotary Positional Embeddings (RoPE) to Q and K projections."""

    def __init__(self):
        """
        Constructs RotaryEmbeddingApplicator object.
        """

        super().__init__()

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embedding_cos: torch.Tensor,
        position_embedding_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rotates query and key states using provided cosine and sine embeddings.

        Args:
            query_states: Query tensor. Shape: `(batch, n_heads, seq_len, head_dim)`.
            key_states: Key tensor. Shape: `(batch, n_kv_heads, seq_len, head_dim)`.
            position_embedding_cos: Cosine values for positions.
                Shape: `(batch, seq_len, head_dim)`.
            position_embedding_sin: Sine values for positions.
                Shape: `(batch, seq_len, head_dim)`.

        Returns:
            A tuple containing the rotated query and key tensors.
        """

        query_states, key_states = _apply_rotary_pos_emb(
            query_states, key_states, position_embedding_cos, position_embedding_sin
        )

        return query_states, key_states
