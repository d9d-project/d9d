from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import nn

from d9d.kernel.cce import linear_cross_entropy
from d9d.module.base import ModuleLateInit

LM_IGNORE_INDEX = -100
"""Index ignored by LM head while calculating logps"""


class SplitLanguageModellingHead(nn.Module, ModuleLateInit):
    """
    A segmented language modeling head that computes per-token cross-entropy loss values using a composed weight matrix.

    This class maintains separate linear layers for different segments of the vocabulary
    (e.g., regular vs. special tokens). During the forward pass, it concatenates the
    weights to form a unified projection matrix and computes the cross-entropy loss
    efficiently, typically using a fused kernel to avoid materializing full logits.

    The concatenation order of the weights is determined by `split_order`, which ensures
    consistency with the global vocabulary indices.
    """

    def __init__(
            self,
            split_vocab_size: dict[str, int],
            split_order: Sequence[str],
            hidden_size: int
    ):
        """
        Constructs the SplitLanguageModellingHead object.

        Args:
            split_vocab_size: A dictionary mapping split names to their output vocabulary sizes.
            split_order: A sequence defining the order in which vocabulary segments should be
                concatenated. This determines the mapping of global indices to specific heads.
            hidden_size: The input dimensionality (hidden state size).
        """

        super().__init__()

        lm_head = nn.ModuleDict({
            split_name: nn.Linear(hidden_size, vocab_size, bias=False)
            for split_name, vocab_size in split_vocab_size.items()
        })

        self.lm_head: Mapping[str, nn.Linear] = cast(Mapping[str, nn.Linear], lm_head)
        self._split_order = split_order
        self._hidden_size = hidden_size

    def forward(
            self,
            hidden_states: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the cross-entropy loss for the given hidden states and labels.

        Args:
            hidden_states: Input tensor of shape `(B, S, H)`.
            labels: Target label tensor of shape `(B, S)`. Indices must correspond
                to the global vocabulary formed by concatenating splits in `split_order`.

        Returns:
            A tensor containing per-token loss values (reduction='none'), matching the
            shape of the labels tensor.
        """

        lm_head_weight = torch.cat([self.lm_head[split_name].weight for split_name in self._split_order], dim=0)

        losses = linear_cross_entropy(
            hidden_states,
            lm_head_weight,
            labels,
            ignore_index=LM_IGNORE_INDEX,
            reduction="none"
        )
        return losses

    def reset_parameters(self):
        """Resets module parameters."""

        for head in self.lm_head.values():
            head.reset_parameters()
