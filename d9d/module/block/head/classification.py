import torch
from torch import nn

from d9d.module.base import ModuleLateInit


class ClassificationHead(nn.Module, ModuleLateInit):
    """
    A classification head module that is typically used on top of model hidden states.

    It applies dropout followed by a linear projection to produce logits for a specified
    number of classes. It supports optional pooling via a mask, allowing for selection
    of specific tokens (e.g., [CLS] tokens or specific sequence positions) before
    projection.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        """
        Constructs the ClassificationHead object.

        Args:
            hidden_size: The input dimensionality (hidden state size).
            num_labels: The number of output classes.
            dropout: The dropout probability.
        """

        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, hidden_states: torch.Tensor, pooling_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Computes class logits from hidden states.

        Args:
            hidden_states: Input tensor of hidden states.
            pooling_mask: Optional mask to select specific hidden states.
                If provided, the input is indexed as `hidden_states[pooling_mask == 1]`,
                flattening the batch and sequence dimensions into a single dimension of
                selected tokens.

        Returns:
            A tensor containing the unnormalized logits.
        """

        if pooling_mask is not None:
            hidden_states = hidden_states[pooling_mask == 1]
        logits = self.score(self.dropout(hidden_states))
        logits = logits.float()  # force convert to FP32 to make sure loss is calculated properly
        return logits

    def reset_parameters(self):
        """Resets module parameters."""
        self.score.reset_parameters()
