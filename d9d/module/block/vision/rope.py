import torch
from torch import nn

from d9d.module.base import ModuleLateInit


class VisionRotaryEmbedding2D(nn.Module, ModuleLateInit):
    """Produces 2D rotary position embeddings for packed vision patches.

    For every patch, half of the rotary frequencies encode its row index and the other half its
    column index. The produced ``(cos, sin)`` tensors follow the HALF layout and plug directly
    into ``RotaryEmbeddingApplicator(style=RotaryEmbeddingStyle.HALF)``.
    """

    def __init__(self, head_dim: int, max_grid_side: int, spatial_merge_size: int, theta: float = 10000.0):
        """Constructs a VisionRotaryEmbedding2D object.

        Args:
            head_dim: The attention head dimension.
            max_grid_side: Maximum supported grid height/width (pre-merge patches) for caching.
            spatial_merge_size: The spatial merge factor used by the patch merger.
            theta: The RoPE base.

        Raises:
            ValueError: If ``head_dim`` is not divisible by 4.
        """
        super().__init__()

        if head_dim % 4 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 4 for 2D rotary embeddings.")

        self._head_dim = head_dim
        self._max_grid_side = max_grid_side
        self._spatial_merge_size = spatial_merge_size
        self._theta = theta

        self.freq_table = nn.Buffer(torch.empty(max_grid_side, head_dim // 4), persistent=False)

    def _position_coords(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Computes per-patch (row, col) coordinates in packed spatial-merge block order.

        Args:
            grid_thw: Per-segment feature grid, shape ``(num_segments, 3)``.

        Returns:
            Coordinates, shape ``(total_patches, 2)``.

        Raises:
            ValueError: If any segment grid side exceeds ``max_grid_side``.
        """
        device = self.freq_table.device
        merge = self._spatial_merge_size

        coords_per_segment = []
        for t, h, w in grid_thw.tolist():
            if h > self._max_grid_side or w > self._max_grid_side:
                raise ValueError(
                    f"Segment grid ({h}, {w}) exceeds the maximum supported grid side ({self._max_grid_side})."
                )

            merged_h, merged_w = h // merge, w // merge

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra = torch.arange(merge, device=device)

            row_idx = block_rows[:, None, None, None] * merge + intra[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge + intra[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge, merge).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge, merge).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            coords_per_segment.append(coords)

        return torch.cat(coords_per_segment)

    def forward(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes packed rotary embeddings for the given segment grids.

        Args:
            grid_thw: Per-segment feature grid (temporal, height, width), shape ``(num_segments, 3)``.

        Returns:
            A tuple of ``(cos, sin)`` tensors, each of shape ``(total_patches, head_dim)``.

        Raises:
            ValueError: If any segment grid side exceeds ``max_grid_side``.
        """
        coords = self._position_coords(grid_thw)

        freqs = self.freq_table[coords].flatten(1)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()

    def reset_parameters(self):
        """Resets module buffer populated values."""
        with torch.no_grad():
            dim = self._head_dim // 2
            inv_freq = 1.0 / (
                self._theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=self.freq_table.device) / dim)
            )
            positions = torch.arange(self._max_grid_side, dtype=torch.float, device=self.freq_table.device)
            self.freq_table.data = torch.outer(positions, inv_freq).to(dtype=self.freq_table.dtype)
