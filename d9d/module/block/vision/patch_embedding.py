import torch
from torch import nn

from d9d.module.base import ModuleLateInit


class PatchEmbedding(nn.Module, ModuleLateInit):
    """Projects packed raw media patches into the vision hidden size.

    Each input row is one flattened spatio-temporal patch
    (``in_channels * temporal_patch_size * patch_size * patch_size`` values); the projection is a
    3D convolution with kernel size equal to stride, i.e. a linear projection per patch.
    """

    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        patch_size: int,
        temporal_patch_size: int,
    ):
        """Constructs a PatchEmbedding object.

        Args:
            hidden_size: The vision encoder hidden size.
            in_channels: Number of input image channels.
            patch_size: Spatial patch size (height and width).
            temporal_patch_size: Temporal patch size (number of frames per patch).
        """
        super().__init__()

        self._in_channels = in_channels
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Projects packed patches.

        Args:
            features: Packed patches, shape ``(total_patches, in_channels * temporal_patch_size *
                patch_size * patch_size)``.

        Returns:
            Patch embeddings, shape ``(total_patches, hidden_size)``.
        """
        target_dtype = self.proj.weight.dtype
        features = features.view(-1, self._in_channels, self._temporal_patch_size, self._patch_size, self._patch_size)
        return self.proj(features.to(dtype=target_dtype)).view(features.shape[0], -1)

    def reset_parameters(self):
        """Resets module parameters."""
        self.proj.reset_parameters()
