from enum import StrEnum

import torch


class TokenPoolingType(StrEnum):
    """Supported token pooling strategies.

    Attributes:
        first: Selects the first token of the sequence, e.g. the ``[CLS]`` token.
        last: Selects the last non-padding token of the sequence, e.g. for decoder-only models.
        all: Selects all non-padding tokens, e.g. for mean pooling.
    """

    first = "first"
    last = "last"
    all = "all"


def token_pooling_mask_from_attention_mask(
    attention_mask: torch.Tensor, pooling_type: TokenPoolingType
) -> torch.Tensor:
    """Builds a binary mask of the tokens to pool for the given strategy.

    ``last`` assumes right padding.

    Args:
        attention_mask: A binary mask of valid tokens (1) and padding (0). Shape: ``(batch, seq_len)``.
        pooling_type: The strategy for selecting tokens.

    Returns:
        A mask with 1 at the positions to pool and 0 elsewhere. ``first`` and ``last`` return a ``torch.long``
        mask. ``all`` returns ``attention_mask`` itself. Shape: ``(batch, seq_len)``.

    Raises:
        ValueError: If ``pooling_type`` is not supported.
    """
    match pooling_type:
        case TokenPoolingType.first:
            mask = torch.zeros_like(attention_mask, dtype=torch.long)
            mask[:, 0] = 1
            return mask
        case TokenPoolingType.last:
            batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
            last_token_indices = attention_mask.sum(dim=1) - 1
            mask = torch.zeros_like(attention_mask, dtype=torch.long)
            mask[batch_indices, last_token_indices] = 1
            return mask
        case TokenPoolingType.all:
            return attention_mask
        case _:
            raise ValueError(f"Unknown pooling type ({pooling_type}).")
