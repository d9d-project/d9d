from collections.abc import Mapping, Sequence
from typing import cast

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
    """Token embedding layer composed of several named, independent embedding tables.

    Each named split (e.g. ``"orig"``, ``"special"``, ``"prompt_prefix"``) owns a contiguous range of global
    vocabulary indices. This is useful for model adaptation, where different sets of tokens need different
    initialization or training behavior.
    """

    def __init__(self, split_vocab_size: dict[str, int], split_order: Sequence[str], hidden_size: int):
        """Constructs the ``SplitTokenEmbeddings`` object.

        Args:
            split_vocab_size: Mapping from split names to their vocabulary sizes.
            split_order: Order in which splits are concatenated to form the global vocabulary.
                Every name must be a key of ``split_vocab_size``.
            hidden_size: Dimensionality of the embedding vectors.
        """
        super().__init__()

        token_embedding = nn.ModuleDict(
            {split_name: nn.Embedding(vocab_size, hidden_size) for split_name, vocab_size in split_vocab_size.items()}
        )
        self.token_embedding: Mapping[str, nn.Embedding] = cast(Mapping[str, nn.Embedding], token_embedding)

        self._id_start, self._id_end = _build_token_start_end_indices(split_vocab_size, split_order)
        self._hidden_size = hidden_size
        self._split_order = split_order

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Looks up embeddings for global vocabulary indices.

        Args:
            input_ids: Global vocabulary indices. Can have any shape.

        Returns:
            Embeddings. Shape: ``(*input_ids.shape, hidden_size)``.

        Raises:
            ValueError: If ``split_order`` is empty.
        """
        output_embeds: torch.Tensor | None = None

        for split_name in self._split_order:
            start_idx = self._id_start[split_name]
            end_idx = self._id_end[split_name]
            layer = self.token_embedding[split_name]
            mask = (input_ids >= start_idx) & (input_ids < end_idx)

            safe_ids = torch.where(mask, input_ids - start_idx, 0)
            masked_embed = layer(safe_ids) * mask[..., None]

            if output_embeds is None:
                output_embeds = masked_embed
            else:
                output_embeds = output_embeds + masked_embed

        if output_embeds is None:
            raise ValueError(f"split_order ({self._split_order}) must contain at least one split.")

        return output_embeds

    def reset_parameters(self):
        """Resets the parameters of all embedding splits."""
        for layer in self.token_embedding.values():
            layer.reset_parameters()
