from collections.abc import Sequence
from enum import StrEnum

import torch
import torch.nn.functional as F


class PaddingSide1D(StrEnum):
    """Side on which 1D sequences are padded.

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
            raise ValueError(f"Unknown padding side ({side}).")


def pad_stack_1d(
    items: Sequence[torch.Tensor],
    pad_value: int,
    padding_side: PaddingSide1D = PaddingSide1D.right,
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    """Pads 1D tensors to the same length and stacks them into a batch.

    All tensors are padded to the length of the longest one.

    Args:
        items: The 1D tensors to stack.
        pad_value: The value used for padding.
        padding_side: The side on which to pad.
        pad_to_multiple_of: If set, the padded length is rounded up to a multiple of this value.

    Returns:
        The stacked tensor. Shape: ``(batch, seq_len)``.

    Raises:
        ValueError: If ``items`` is empty or ``pad_to_multiple_of`` is not positive.
    """
    if not items:
        raise ValueError("Cannot stack 0 items. Pass at least one tensor.")
    if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
        raise ValueError(f"pad_to_multiple_of ({pad_to_multiple_of}) must be positive.")

    max_len = max(x.shape[0] for x in items)

    if pad_to_multiple_of is not None and (remainder := max_len % pad_to_multiple_of) != 0:
        max_len = max_len + (pad_to_multiple_of - remainder)

    padded_items = []

    for x in items:
        difference = max_len - x.shape[0]

        if difference == 0:
            padded_items.append(x)
        else:
            padded_items.append(F.pad(x, _padding_side_1d_to_config(padding_side, difference), value=pad_value))

    return torch.stack(padded_items, dim=0)
