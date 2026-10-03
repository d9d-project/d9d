from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import nn

from d9d.kernel.cce import linear_cross_entropy
from d9d.module.block.head.base import TaskHead
from d9d.module.block.head.io import SequenceCausalLMHeadShared, SequenceCausalLMOutput

LM_IGNORE_INDEX = -100
"""Label index that the LM head ignores when computing the loss."""


class SplitLanguageModellingHead(TaskHead[SequenceCausalLMHeadShared, SequenceCausalLMOutput]):
    """Language modeling head with a split vocabulary that computes per-token cross-entropy loss.

    Each vocabulary split (e.g. regular and special tokens) has its own linear layer. The forward pass
    concatenates their weights in ``split_order`` and computes the loss with the fused Cut Cross-Entropy
    kernel, so full logits are not materialized. Requires the ``d9d[cce]`` extra.
    """

    def __init__(self, split_vocab_size: Mapping[str, int], split_order: Sequence[str], hidden_size: int):
        """Constructs the ``SplitLanguageModellingHead`` object.

        Args:
            split_vocab_size: Mapping from split names to their vocabulary sizes.
            split_order: Order in which vocabulary splits are concatenated. This defines which global
                indices belong to each split.
            hidden_size: Hidden size.
        """
        super().__init__()

        lm_head = nn.ModuleDict(
            {
                split_name: nn.Linear(hidden_size, vocab_size, bias=False)
                for split_name, vocab_size in split_vocab_size.items()
            }
        )

        self.lm_head: Mapping[str, nn.Linear] = cast(Mapping[str, nn.Linear], lm_head)
        self._split_order = split_order
        self._hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor, shared: SequenceCausalLMHeadShared) -> SequenceCausalLMOutput:
        """Computes the cross-entropy loss for the given hidden states and labels.

        Args:
            hidden_states: Hidden states. Shape: ``(batch, seq_len, hidden_size)``.
            shared: The head shared input. Its ``labels`` index the global vocabulary formed by
                concatenating the splits in ``split_order``. Labels equal to ``LM_IGNORE_INDEX`` are ignored.

        Returns:
            The causal LM output holding the unreduced per-token loss, with the shape of ``labels``.
        """
        lm_head_weight = torch.cat([self.lm_head[split_name].weight for split_name in self._split_order], dim=0)

        losses = linear_cross_entropy(
            hidden_states, lm_head_weight, shared.labels, ignore_index=LM_IGNORE_INDEX, reduction="none"
        )
        return SequenceCausalLMOutput(logps=losses)

    def reset_parameters(self):
        """Resets module parameters."""
        for head in self.lm_head.values():
            head.reset_parameters()
