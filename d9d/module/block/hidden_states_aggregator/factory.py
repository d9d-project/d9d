from enum import StrEnum

import torch

from .base import BaseHiddenStatesAggregator
from .mean import HiddenStatesAggregatorMean
from .noop import HiddenStatesAggregatorNoOp


class HiddenStatesAggregationMode(StrEnum):
    """Enumeration of available hidden state aggregation strategies.

    Attributes:
        no: Performs no aggregation (No-Op).
        mean: Computes the mean of hidden states, taking a mask into account.
    """

    no = "no"
    mean = "mean"


def create_hidden_states_aggregator(
    mode: HiddenStatesAggregationMode, agg_mask: torch.Tensor | None
) -> BaseHiddenStatesAggregator:
    """Factory function to create a hidden states aggregator.

    Args:
        mode: The specific aggregation mode to instantiate.
        agg_mask: A tensor mask required for specific modes.
            Can be None if the selected mode does not require masking.

    Returns:
        An instance of a concrete BaseHiddenStatesAggregator subclass.

    Raises:
        ValueError: If 'mean' mode is selected but 'agg_mask' is None, or if
            an unknown mode is provided.
    """

    match mode:
        case HiddenStatesAggregationMode.no:
            return HiddenStatesAggregatorNoOp()
        case HiddenStatesAggregationMode.mean:
            if agg_mask is None:
                raise ValueError("You have to specify aggregation mask")
            return HiddenStatesAggregatorMean(agg_mask)
        case _:
            raise ValueError("Unknown hidden states aggregation mode")
