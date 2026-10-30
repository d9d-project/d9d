import torch
from torch import nn

from d9d.module.block.head.base import TaskHead
from d9d.module.block.head.io import SequenceClassificationOutput, SequencePoolingHeadShared


class ClassificationHead(TaskHead[SequencePoolingHeadShared, SequenceClassificationOutput]):
    """Classification head on top of model hidden states.

    Applies dropout and a linear projection to produce logits for ``num_labels`` classes.
    An optional pooling mask selects specific tokens (e.g. ``[CLS]`` tokens) before the projection.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        """Constructs the ``ClassificationHead`` object.

        Args:
            hidden_size: Hidden size.
            num_labels: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, hidden_states: torch.Tensor, shared: SequencePoolingHeadShared) -> SequenceClassificationOutput:
        """Computes class logits from hidden states.

        Args:
            hidden_states: Hidden states. Shape: ``(batch, seq_len, hidden_size)``.
            shared: The head shared input. Its optional ``pooling_mask`` selects hidden states as
                ``hidden_states[pooling_mask == 1]``, which flattens the batch and sequence dimensions.

        Returns:
            The classification output holding the unnormalized fp32 logits.
        """
        if shared.pooling_mask is not None:
            hidden_states = hidden_states[shared.pooling_mask == 1]
        logits = self.score(self.dropout(hidden_states))
        # Upcast to fp32 so the loss is computed in full precision.
        logits = logits.float()
        return SequenceClassificationOutput(scores=logits)

    def reset_parameters(self):
        """Resets module parameters."""
        self.score.reset_parameters()
