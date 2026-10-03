import torch
from torch import nn

from d9d.module.base import ModuleLateInit


class InterpolatedPositionEmbedding(nn.Module, ModuleLateInit):
    """Learned absolute position embeddings bilinearly interpolated to each segment's grid.

    The embedding table represents a fixed square grid; for every media segment the table is
    bilinearly resampled to the segment's ``(height, width)`` feature grid, repeated across its
    temporal frames, and reordered into spatial-merge block order (all patches of one merge block
    are contiguous) to match the packing produced by the patch merger.
    """

    def __init__(self, hidden_size: int, num_position_embeddings: int, spatial_merge_size: int):
        """Constructs an InterpolatedPositionEmbedding object.

        Args:
            hidden_size: The vision encoder hidden size.
            num_position_embeddings: Total number of positions in the learned table. Must be a
                perfect square.
            spatial_merge_size: The spatial merge factor used by the patch merger.

        Raises:
            ValueError: If ``num_position_embeddings`` is not a perfect square.
        """
        super().__init__()

        num_grid_per_side = int(num_position_embeddings**0.5)
        if num_grid_per_side * num_grid_per_side != num_position_embeddings:
            raise ValueError(f"num_position_embeddings ({num_position_embeddings}) must be a perfect square.")

        self._num_grid_per_side = num_grid_per_side
        self._spatial_merge_size = spatial_merge_size

        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)

    def _interpolate_one(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        side = self._num_grid_per_side

        h_idxs = torch.linspace(0, side - 1, height, device=device)
        w_idxs = torch.linspace(0, side - 1, width, device=device)

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clip(max=side - 1)
        w_ceil = (w_floor + 1).clip(max=side - 1)

        dh = (h_idxs - h_floor)[:, None]
        dw = (w_idxs - w_floor)[None, :]

        indices = torch.stack(
            [
                (h_floor[:, None] * side + w_floor[None, :]).flatten(),
                (h_floor[:, None] * side + w_ceil[None, :]).flatten(),
                (h_ceil[:, None] * side + w_floor[None, :]).flatten(),
                (h_ceil[:, None] * side + w_ceil[None, :]).flatten(),
            ]
        )
        weights = torch.stack(
            [
                ((1 - dh) * (1 - dw)).flatten(),
                ((1 - dh) * dw).flatten(),
                (dh * (1 - dw)).flatten(),
                (dh * dw).flatten(),
            ]
        ).to(self.pos_embed.weight.dtype)

        embeds = self.pos_embed(indices) * weights[:, :, None]
        return embeds.sum(dim=0)

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Computes packed position embeddings for the given segment grids.

        Args:
            grid_thw: Per-segment feature grid (temporal, height, width), shape ``(num_segments, 3)``.

        Returns:
            Position embeddings, shape ``(total_patches, hidden_size)``, in packed
            spatial-merge block order.
        """
        device = self.pos_embed.weight.device
        merge = self._spatial_merge_size

        segments = []
        for t, h, w in grid_thw.tolist():
            embeds = self._interpolate_one(h, w, device)
            embeds = embeds.repeat(t, 1)
            embeds = embeds.view(t, h // merge, merge, w // merge, merge, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            segments.append(embeds)

        return torch.cat(segments)

    def reset_parameters(self):
        """Resets module parameters."""
        self.pos_embed.reset_parameters()
