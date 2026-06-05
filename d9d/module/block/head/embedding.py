import torch
import torch.nn.functional as F
from torch import nn

from d9d.module.base import ModuleLateInit


class EmbeddingHead(nn.Module, ModuleLateInit):
    """A head module for extracting dense representations from hidden states.

    It optionally applies a linear projection and L2 normalization to produce
    embeddings for contrastive learning or retrieval tasks. It supports boolean
    masking to select specific tokens (e.g., the last token) before extraction.
    """

    def __init__(self, hidden_size: int, embedding_dim: int | None, normalize: bool):
        """Constructs the EmbeddingHead object.

        Args:
            hidden_size: The input dimensionality (hidden state size).
            embedding_dim: The dimensionality of the output embedding. If None,
                additional linear projection won't be applied.
            normalize: Whether to apply L2 normalization to the final embeddings.
        """
        super().__init__()

        self._normalize = normalize

        if embedding_dim is not None:
            self.projection = nn.Linear(hidden_size, embedding_dim, bias=False)
        else:
            self.projection = None

    def forward(self, hidden_states: torch.Tensor, pooling_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Computes dense embeddings from hidden states.

        Args:
            hidden_states: Input tensor of hidden states.
            pooling_mask: Optional mask to select specific hidden states.
                If provided, the input is indexed as `hidden_states[pooling_mask == 1]`,
                flattening the batch and sequence dimensions into a single dimension of
                selected tokens.

        Returns:
            A tensor containing embeddings.
        """
        if pooling_mask is not None:
            hidden_states = hidden_states[pooling_mask == 1]

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)

        # convert to fp32 before normalization for numerical stability
        hidden_states = hidden_states.float()

        if self._normalize:
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)

        return hidden_states

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        if self.projection is not None:
            self.projection.reset_parameters()
