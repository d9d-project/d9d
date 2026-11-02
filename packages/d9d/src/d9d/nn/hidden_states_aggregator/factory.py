from enum import StrEnum

import torch

from .base import BaseHiddenStatesAggregator
from .mean import HiddenStatesAggregatorMean
from .noop import HiddenStatesAggregatorNoOp


class HiddenStatesAggregationMode(StrEnum):
    """Available hidden states aggregation strategies.

    Attributes:
        no: Performs no aggregation.
        mean: Computes the masked mean of hidden states over the sequence.
    """

    no = "no"
    mean = "mean"


def create_hidden_states_aggregator(
    mode: HiddenStatesAggregationMode, agg_mask: torch.Tensor | None
) -> BaseHiddenStatesAggregator:
    """Creates a hidden states aggregator.

    Args:
        mode: Aggregation mode.
        agg_mask: Aggregation mask. Required for the ``mean`` mode, can be ``None`` otherwise.
            Shape: ``(batch, seq_len)``.

    Returns:
        The hidden states aggregator.

    Raises:
        ValueError: If ``mode`` is ``mean`` and ``agg_mask`` is ``None``, or if ``mode`` is unknown.
    """
    match mode:
        case HiddenStatesAggregationMode.no:
            return HiddenStatesAggregatorNoOp()
        case HiddenStatesAggregationMode.mean:
            if agg_mask is None:
                raise ValueError("The mean aggregation mode requires an aggregation mask, but agg_mask is None.")
            return HiddenStatesAggregatorMean(agg_mask)
        case _:
            raise ValueError(f"Unknown hidden states aggregation mode ({mode}).")
