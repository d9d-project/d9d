import torch

from .base import BaseHiddenStatesAggregator


class HiddenStatesAggregatorNoOp(BaseHiddenStatesAggregator):
    """Aggregator that does nothing.

    Used when aggregation is disabled in the configuration.
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
            Always ``None``, even if a snapshot is given.
        """
