from enum import StrEnum

import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.positional.rope_scaling import NoRopeScaling, RopeScaling


class RotaryEmbeddingStyle(StrEnum):
    """Supported rotary position embedding (RoPE) layout styles.

    Attributes:
        HALF: Rotates pairs formed by the first and second halves of the feature dimension.
        INTERLEAVED: Rotates pairs of adjacent feature elements.
    """

    HALF = "half"
    INTERLEAVED = "interleaved"


def prepare_rotary_cos_sin_emb(
    rope_base: int,
    head_dim: int,
    max_position_ids: int,
    device: torch.device,
    dtype: torch.dtype,
    style: RotaryEmbeddingStyle,
    rope_scaling: RopeScaling | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes rotary cosine and sine embeddings.

    Args:
        rope_base: Base of the geometric progression of RoPE frequencies.
        head_dim: Dimensionality of the attention head.
        max_position_ids: Number of positions to precompute.
        device: Target device for the tensors.
        dtype: Target data type for the tensors.
        style: RoPE layout style.
        rope_scaling: Optional scaling strategy. If ``None``, ``NoRopeScaling`` is used.

    Returns:
        A tuple of cosine and sine tensors. Shape of each: ``(max_position_ids, head_dim)``.

    Raises:
        ValueError: If the RoPE style is unknown.
    """
    if rope_scaling is None:
        rope_scaling = NoRopeScaling()

    position_ids = torch.arange(0, max_position_ids, dtype=torch.long)
    freqs = rope_scaling.inverse_frequencies(rope_base, head_dim)

    arguments = (freqs[:, None] @ position_ids[None, :].float()).T

    match style:
        case RotaryEmbeddingStyle.HALF:
            emb = torch.cat((arguments, arguments), dim=-1)
        case RotaryEmbeddingStyle.INTERLEAVED:
            emb = torch.repeat_interleave(arguments, 2, dim=-1)
        case _:
            raise ValueError(f"Unknown RoPE style ({style}).")

    cos = emb.cos()
    sin = emb.sin()

    mscale = rope_scaling.attention_mscale
    cos = cos * mscale
    sin = sin * mscale

    return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)


class RotaryEmbeddingProvider(nn.Module, ModuleLateInit):
    """Module that caches rotary position embeddings and returns them for given positions."""

    def __init__(
        self,
        rope_base: int,
        head_dim: int,
        max_position_ids: int,
        style: RotaryEmbeddingStyle,
        rope_scaling: RopeScaling | None = None,
    ) -> None:
        """Constructs the ``RotaryEmbeddingProvider`` object.

        Args:
            rope_base: Base of the geometric progression of RoPE frequencies.
            head_dim: Dimensionality of the attention head.
            max_position_ids: Number of cached positions. Position indices must be smaller than this value.
            style: RoPE layout style.
            rope_scaling: Optional scaling strategy for extended context lengths. If ``None``,
                ``NoRopeScaling`` is used.
        """
        super().__init__()
        self._rope_base = rope_base
        self._head_dim = head_dim
        self._max_position_ids = max_position_ids
        self._style = style
        self._rope_scaling: RopeScaling = rope_scaling if rope_scaling is not None else NoRopeScaling()
        self.cos_emb = nn.Buffer(torch.empty(max_position_ids, head_dim), persistent=False)
        self.sin_emb = nn.Buffer(torch.empty(max_position_ids, head_dim), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the cached cosine and sine embeddings for the given positions.

        Args:
            position_ids: Position indices, usually with shape ``(batch, seq_len)``.

        Returns:
            A tuple of ``(cos, sin)`` tensors. Shape of each: ``(*position_ids.shape, head_dim)``.
        """
        return self.cos_emb[position_ids], self.sin_emb[position_ids]

    def reset_parameters(self) -> None:
        """Recomputes the cached cosine and sine buffers."""
        with torch.no_grad():
            cos, sin = prepare_rotary_cos_sin_emb(
                rope_base=self._rope_base,
                head_dim=self._head_dim,
                max_position_ids=self._max_position_ids,
                device=self.cos_emb.device,
                dtype=self.cos_emb.dtype,
                style=self._style,
                rope_scaling=self._rope_scaling,
            )
            self.cos_emb.data = cos
            self.sin_emb.data = sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half-chunked elements.

    Returns:
        The rotated tensor.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """Rotates interleaved complex pairs.

    Returns:
        The rotated tensor.
    """
    x_unflattened = x.view(*x.shape[:-1], -1, 2)
    x1 = x_unflattened[..., 0]
    x2 = x_unflattened[..., 1]
    return torch.stack((-x2, x1), dim=-1).view(*x.shape)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    style: RotaryEmbeddingStyle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to ``q`` and ``k``.

    Returns:
        A tuple of rotated ``(q, k)`` tensors.

    Raises:
        ValueError: If the RoPE style is unknown.
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    match style:
        case RotaryEmbeddingStyle.HALF:
            rotate_fn = _rotate_half
        case RotaryEmbeddingStyle.INTERLEAVED:
            rotate_fn = _rotate_every_two
        case _:
            raise ValueError(f"Unknown RoPE style ({style}).")

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)
    return q_embed, k_embed


class RotaryEmbeddingApplicator(nn.Module):
    """Applies rotary position embeddings (RoPE) to Q and K projections."""

    def __init__(self, style: RotaryEmbeddingStyle) -> None:
        """Constructs the ``RotaryEmbeddingApplicator`` object.

        Args:
            style: RoPE layout style.
        """
        super().__init__()
        self._style = style

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embedding_cos: torch.Tensor,
        position_embedding_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotates query and key states using provided cosine and sine embeddings.

        Args:
            query_states: Query tensor. Shape: ``(batch, seq_len, num_heads, head_dim)``.
            key_states: Key tensor. Shape: ``(batch, seq_len, num_kv_heads, head_dim)``.
            position_embedding_cos: Cosine values for the positions. Shape: ``(batch, seq_len, head_dim)``.
            position_embedding_sin: Sine values for the positions. Shape: ``(batch, seq_len, head_dim)``.

        Returns:
            A tuple of the rotated query and key tensors, with the input shapes.
        """
        query_states, key_states = _apply_rotary_pos_emb(
            query_states, key_states, position_embedding_cos, position_embedding_sin, style=self._style
        )

        return query_states, key_states
