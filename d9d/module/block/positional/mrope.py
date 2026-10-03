import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.positional.rope import RotaryEmbeddingStyle, prepare_rotary_cos_sin_emb
from d9d.module.block.positional.rope_scaling import RopeScaling


def _build_plane_selector(mrope_section: tuple[int, int, int], rope_dim: int, interleaved: bool) -> torch.Tensor:
    """Builds the per-frequency plane selector for multimodal rotary embeddings.

    Args:
        mrope_section: Number of frequency pairs allocated to the (temporal, height, width) planes.
        rope_dim: The rotary dimension (HALF layout, i.e. twice the number of frequency pairs).
        interleaved: If True, uses the interleaved ``[THWTHW...TT]`` frequency layout; otherwise
            the chunked ``[TT..HH..WW]`` layout.

    Returns:
        An integer tensor of shape ``(rope_dim,)`` holding the plane index (0=T, 1=H, 2=W) for
        every rotary feature.
    """
    half_dim = rope_dim // 2
    selector = torch.zeros(half_dim, dtype=torch.long)

    if interleaved:
        for plane in (1, 2):
            length = mrope_section[plane] * 3
            selector[plane:length:3] = plane
    else:
        offset = mrope_section[0]
        selector[offset : offset + mrope_section[1]] = 1
        selector[offset + mrope_section[1] : offset + mrope_section[1] + mrope_section[2]] = 2

    # The HALF layout duplicates every frequency across the two halves of the feature dim.
    return torch.cat([selector, selector])


class MultimodalRotaryEmbeddingProvider(nn.Module, ModuleLateInit):
    """Module that manages and provides Multimodal Rotary Positional Embeddings (MRoPE).

    Consumes 3D position ids — separate position planes for the temporal, height and width axes —
    and combines per-plane rotary embeddings according to ``mrope_section``: every rotary
    frequency is assigned to exactly one plane.

    For text tokens, whose position ids are identical across the three planes, the output is
    numerically identical to the standard ``RotaryEmbeddingProvider``.
    """

    def __init__(
        self,
        rope_base: int,
        rope_dim: int,
        max_position_ids: int,
        mrope_section: tuple[int, int, int],
        interleaved: bool = True,
        rope_scaling: RopeScaling | None = None,
    ) -> None:
        """Constructs the MultimodalRotaryEmbeddingProvider.

        Args:
            rope_base: Base geometrical progression period for RoPE.
            rope_dim: The rotary dimension (equal to the attention head dim, or smaller when
                partial RoPE is used).
            max_position_ids: Maximum supported position index for caching.
            mrope_section: Number of frequency pairs allocated to the (temporal, height, width)
                position planes. Must sum to ``rope_dim // 2``.
            interleaved: If True, uses the interleaved ``[THWTHW...TT]`` frequency layout
                (Qwen3-VL style); otherwise the chunked ``[TT..HH..WW]`` layout (Qwen2-VL style).
            rope_scaling: Optional scaling configuration for extended context lengths.

        Raises:
            ValueError: If ``mrope_section`` does not sum to ``rope_dim // 2``.
        """
        super().__init__()

        if sum(mrope_section) != rope_dim // 2:
            raise ValueError(f"mrope_section {mrope_section} must sum to rope_dim // 2 ({rope_dim // 2}).")

        self._rope_base = rope_base
        self._rope_dim = rope_dim
        self._max_position_ids = max_position_ids
        self._mrope_section = mrope_section
        self._interleaved = interleaved
        self._rope_scaling = rope_scaling

        self.cos_emb = nn.Buffer(torch.empty(max_position_ids, rope_dim), persistent=False)
        self.sin_emb = nn.Buffer(torch.empty(max_position_ids, rope_dim), persistent=False)
        self.plane_selector = nn.Buffer(_build_plane_selector(mrope_section, rope_dim, interleaved), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves combined cosine and sine embeddings for 3D positions.

        Args:
            position_ids: Tensor of position indices, shape ``(3, batch, seq)`` — the temporal,
                height and width position planes.

        Returns:
            A tuple of (cos, sin) tensors, each of shape ``(batch, seq, rope_dim)``.

        Raises:
            ValueError: If ``position_ids`` does not have three position planes.
        """
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(f"position_ids must have shape (3, batch, seq), got {tuple(position_ids.shape)}.")

        cos_planes = self.cos_emb[position_ids]  # (3, batch, seq, rope_dim)
        sin_planes = self.sin_emb[position_ids]

        selector = self.plane_selector.view(1, 1, -1).expand(*position_ids.shape[1:], -1)  # (batch, seq, rope_dim)

        cos = cos_planes.gather(dim=0, index=selector.unsqueeze(0)).squeeze(0)
        sin = sin_planes.gather(dim=0, index=selector.unsqueeze(0)).squeeze(0)

        return cos, sin

    def reset_parameters(self) -> None:
        """Resets module buffer populated values."""
        with torch.no_grad():
            cos, sin = prepare_rotary_cos_sin_emb(
                rope_base=self._rope_base,
                head_dim=self._rope_dim,
                max_position_ids=self._max_position_ids,
                device=self.cos_emb.device,
                dtype=self.cos_emb.dtype,
                style=RotaryEmbeddingStyle.HALF,
                rope_scaling=self._rope_scaling,
            )
            self.cos_emb.data = cos
            self.sin_emb.data = sin
            self.plane_selector.data = _build_plane_selector(self._mrope_section, self._rope_dim, self._interleaved).to(
                self.plane_selector.device
            )
