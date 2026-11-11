import torch

from .base import BaseHiddenStatesAggregator


class HiddenStatesAggregatorNoOp(BaseHiddenStatesAggregator):
    """Aggregator that does nothing.

    Used when aggregation is disabled in the configuration.
    """

    def collect(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Keeps nothing.

        Args:
            hidden_states: Ignored.

        Returns:
            An empty tuple.
        """
        return ()

    def pack(self, collected: tuple[torch.Tensor, ...], snapshot: torch.Tensor | None) -> torch.Tensor | None:
        """Packs nothing.

        Args:
            collected: Ignored.
            snapshot: Ignored.

        Returns:
            Always ``None``, even if a snapshot is given.
        """
