from collections.abc import Sequence
from enum import StrEnum

import torch
import torch.nn.functional as F


class PaddingSide1D(StrEnum):
    """
    Enum specifying the side for padding 1D sequences.

    Attributes:
        left: Pad on the left side.
        right: Pad on the right side.
    """

    left = "left"
    right = "right"


def _padding_side_1d_to_config(side: PaddingSide1D, difference: int) -> tuple[int, ...]:
    match side:
        case PaddingSide1D.left:
            return difference, 0
        case PaddingSide1D.right:
            return 0, difference
        case _:
            raise ValueError("Unknown padding side")


def pad_stack_1d(
        items: Sequence[torch.Tensor],
        pad_value: int,
        padding_side: PaddingSide1D = PaddingSide1D.right,
        pad_to_multiple_of: int | None = None
) -> torch.Tensor:
    """
    Stacks 1D tensors into a batch, applying padding.

    Calculates the maximum length among the input tensors (optionally aligning to a multiple),
    pads elements to match this length on the specified side, and stacks them.

    Args:
        items: A sequence of 1D tensors to be stacked.
        pad_value: The value used for padding.
        padding_side: The side on which to apply padding (left or right).
        pad_to_multiple_of: Optional integer. If provided, ensures the target length
            is a multiple of this value.

    Returns:
        A single stacked tensor of shape (batch, max_length).

    Raises:
        ValueError: If no items are provided or if `pad_to_multiple_of` is <= 0.
    """

    if not items:
        raise ValueError("Cannot stack 0 items")
    if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
        raise ValueError("pad_to_multiple_of should be > 0")

    max_len = max(x.shape[0] for x in items)

    if pad_to_multiple_of is not None and (remainder := max_len % pad_to_multiple_of) != 0:
        max_len = max_len + (pad_to_multiple_of - remainder)

    padded_items = []

    for x in items:
        difference = max_len - x.shape[0]

        if difference == 0:
            padded_items.append(x)
        else:
            padded_items.append(
                F.pad(x, _padding_side_1d_to_config(padding_side, difference), value=pad_value)
            )

    return torch.stack(padded_items, dim=0)
