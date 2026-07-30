import torch
import torch.nn.functional as F
from torch import nn

from d9d.module.base import ModuleLateInit


class SpatialPatchMerger(nn.Module, ModuleLateInit):
    """Merges neighboring patch embeddings and projects them to the language model hidden size.

    Consumes packed patch embeddings in spatial-merge block order (all ``merge_size**2`` patches
    of one merge block are contiguous), concatenates each block along the feature dimension, and
    applies ``LayerNorm -> Linear -> GELU -> Linear``.
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        norm_eps: float,
    ):
        """Constructs a SpatialPatchMerger object.

        Args:
            hidden_size: The vision encoder hidden size.
            out_hidden_size: The output (language model) hidden size.
            spatial_merge_size: The spatial merge factor; each output token merges
                ``spatial_merge_size**2`` patches.
            norm_eps: Epsilon value for the layer normalization.
        """
        super().__init__()

        self._merged_hidden_size = hidden_size * (spatial_merge_size**2)

        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.fc1 = nn.Linear(self._merged_hidden_size, self._merged_hidden_size)
        self.fc2 = nn.Linear(self._merged_hidden_size, out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Merges and projects packed patch embeddings.

        Args:
            hidden_states: Packed patch embeddings, shape ``(total_patches, hidden_size)``, in
                spatial-merge block order.

        Returns:
            Merged media token embeddings, shape
            ``(total_patches / spatial_merge_size**2, out_hidden_size)``.
        """
        hidden_states = self.norm(hidden_states).view(-1, self._merged_hidden_size)
        return self.fc2(F.gelu(self.fc1(hidden_states)))

    def reset_parameters(self):
        """Resets module parameters."""
        self.norm.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
