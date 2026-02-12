from enum import StrEnum

import torch


class TokenPoolingType(StrEnum):
    """Enumeration of supported token pooling strategies.

    Attributes:
        first: Selects the first token of the sequence (e.g., [CLS] token).
        last: Selects the last non-padding token of the sequence (e.g., for Transformer Decoder).
        all: Selects all non-padding tokens (e.g., for mean pooling).
    """

    first = "first"
    last = "last"
    all = "all"


def token_pooling_mask_from_attention_mask(
    attention_mask: torch.Tensor, pooling_type: TokenPoolingType
) -> torch.Tensor:
    """Generates a binary mask for token pooling based on the specified strategy.

    Args:
        attention_mask: A binary mask indicating valid tokens (1) and padding (0).
            Expected shape is (batch_size, sequence_length).
        pooling_type: The strategy to use for selecting tokens.

    Returns:
        A LongTensor of the same shape as input containing 1s at positions
        to be included in pooling and 0s elsewhere.

    Raises:
        ValueError: If the provided pooling type is not supported.
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
            raise ValueError(f"Unknown pooling type: {pooling_type}")
