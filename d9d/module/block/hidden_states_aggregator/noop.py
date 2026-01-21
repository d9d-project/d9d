import torch

from .base import BaseHiddenStatesAggregator


class HiddenStatesAggregatorNoOp(BaseHiddenStatesAggregator):
    """Aggregator implementation that performs no operations.

    This acts as a null object for cases where aggregation is disabled in the configuration.
    """

    def add_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Does nothing.

        Args:
            hidden_states: Ignored.
        """

    def pack_with_snapshot(self, snapshot: torch.Tensor | None) -> torch.Tensor | None:
        """Does nothing.

        Args:
            snapshot: Ignored.

        Returns:
            None.
        """
