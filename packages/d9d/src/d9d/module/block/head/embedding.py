import torch
import torch.nn.functional as F
from torch import nn

from d9d.module.block.head.base import TaskHead
from d9d.module.block.head.io import SequenceEmbeddingOutput, SequencePoolingHeadShared


class EmbeddingHead(TaskHead[SequencePoolingHeadShared, SequenceEmbeddingOutput]):
    """Head that extracts dense embeddings from hidden states.

    Optionally applies a linear projection and L2 normalization, e.g. for contrastive learning or retrieval.
    An optional pooling mask selects specific tokens (e.g. the last token) before the projection.
    """

    def __init__(self, hidden_size: int, embedding_dim: int | None, normalize: bool):
        """Constructs the ``EmbeddingHead`` object.

        Args:
            hidden_size: Hidden size.
            embedding_dim: Dimensionality of the output embedding. If ``None``, no linear projection is applied.
            normalize: Whether to apply L2 normalization to the final embeddings.
        """
        super().__init__()

        self._normalize = normalize

        if embedding_dim is not None:
            self.projection = nn.Linear(hidden_size, embedding_dim, bias=False)
        else:
            self.projection = None

    def forward(self, hidden_states: torch.Tensor, shared: SequencePoolingHeadShared) -> SequenceEmbeddingOutput:
        """Computes dense embeddings from hidden states.

        Args:
            hidden_states: Hidden states. Shape: ``(batch, seq_len, hidden_size)``.
            shared: The head shared input. Its optional ``pooling_mask`` selects hidden states as
                ``hidden_states[pooling_mask == 1]``, which flattens the batch and sequence dimensions.

        Returns:
            The embedding output holding the fp32 embeddings.
        """
        if shared.pooling_mask is not None:
            hidden_states = hidden_states[shared.pooling_mask == 1]

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)

        # Upcast to fp32 before normalization for numerical stability.
        hidden_states = hidden_states.float()

        if self._normalize:
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)

        return SequenceEmbeddingOutput(embeddings=hidden_states)

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        if self.projection is not None:
            self.projection.reset_parameters()
