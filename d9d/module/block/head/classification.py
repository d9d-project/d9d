import torch
from torch import nn

from d9d.module.block.head.base import TaskHead
from d9d.module.block.head.io import SequenceClassificationOutput, SequencePoolingHeadShared


class ClassificationHead(TaskHead[SequencePoolingHeadShared, SequenceClassificationOutput]):
    """A classification head module that is typically used on top of model hidden states.

    It applies dropout followed by a linear projection to produce logits for a specified
    number of classes. It supports optional pooling via a mask, allowing for selection
    of specific tokens (e.g., [CLS] tokens or specific sequence positions) before
    projection.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        """Constructs the ClassificationHead object.

        Args:
            hidden_size: The input dimensionality (hidden state size).
            num_labels: The number of output classes.
            dropout: The dropout probability.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, hidden_states: torch.Tensor, shared: SequencePoolingHeadShared) -> SequenceClassificationOutput:
        """Computes class logits from hidden states.

        Args:
            hidden_states: Input tensor of hidden states.
            shared: The head shared input. Its optional `pooling_mask` selects specific hidden
                states: the input is indexed as `hidden_states[pooling_mask == 1]`, flattening the
                batch and sequence dimensions into a single dimension of selected tokens.

        Returns:
            The classification output holding the unnormalized logits.
        """
        if shared.pooling_mask is not None:
            hidden_states = hidden_states[shared.pooling_mask == 1]
        logits = self.score(self.dropout(hidden_states))
        logits = logits.float()  # force convert to FP32 to make sure loss is calculated properly
        return SequenceClassificationOutput(scores=logits)

    def reset_parameters(self):
        """Resets module parameters."""
        self.score.reset_parameters()
