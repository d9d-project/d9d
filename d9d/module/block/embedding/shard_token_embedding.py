from typing import Sequence

import torch
from torch import nn

from d9d.module.base import ModuleLateInit


def _build_token_start_end_indices(
        split_vocab_size: dict[str, int], split_order: Sequence[str]
) -> tuple[dict[str, int], dict[str, int]]:
    offset = 0
    starts = {}
    ends = {}
    for split in split_order:
        current_size = split_vocab_size[split]

        starts[split] = offset
        ends[split] = offset + current_size

        offset += current_size
    return starts, ends


class SplitTokenEmbeddings(nn.Module, ModuleLateInit):
    """
    A token embedding layer composed of multiple named, independent embedding tables.

    This class maintains a dictionary of embedding layers, mapping contiguous
    ranges of global vocabulary indices to specific named splits (e.g., 'orig',
    'special', 'prompt_prefix'). This is useful for model adaptation strategies where
    different sets of tokens require different initialization  training behaviors.
    """

    def __init__(
            self,
            split_vocab_size: dict[str, int],
            split_order: Sequence[str],
            hidden_size: int
    ):
        """
        Constructs the SplitTokenEmbeddings object.

        Args:
            split_vocab_size: A dictionary mapping split names to their vocabulary sizes.
            split_order: A sequence defining the order in which splits are concatenated
                to form the global vocabulary. Keys provided here must exist in
                split_vocab_size.
            hidden_size: The dimensionality of the embedding vectors.
        """

        super().__init__()

        self.token_embedding = nn.ModuleDict({
            split_name: nn.Embedding(vocab_size, hidden_size)
            for split_name, vocab_size in split_vocab_size.items()
        })

        self._id_start, self._id_end = _build_token_start_end_indices(split_vocab_size, split_order)
        self._hidden_size = hidden_size
        self._split_order = split_order

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieves embeddings for the input indices by routing them to appropriate internal layers.

        Args:
            input_ids: Tensor of arbitrary shape containing global vocabulary indices.

        Returns:
            Tensor of same shape as input_ids plus a last dimension of hidden_size.
        """

        metadata_weight = next(iter(self.token_embedding.values())).weight
        # todo custom cuda kernel for indexing and filling?

        embed = torch.empty(
            size=(input_ids.shape[0], input_ids.shape[1], self._hidden_size),
            device=metadata_weight.device,
            dtype=metadata_weight.dtype,
        )

        for split_name in self._split_order:
            start_idx, end_idx = self._id_start[split_name], self._id_end[split_name]
            is_split_mask = (input_ids >= start_idx) & (input_ids < end_idx)
            split_embed = self.token_embedding[split_name](input_ids[is_split_mask] - start_idx)
            embed[is_split_mask] = split_embed

        return embed

    def reset_parameters(self):
        """
        Resets parameters for all registered embedding splits.
        """

        for layer in self.token_embedding.values():
            layer.reset_parameters()
