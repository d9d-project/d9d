import abc

import torch


class BaseHiddenStatesAggregator(abc.ABC):
    """Abstract base class for hidden states aggregation strategies.

    This interface defines how hidden states should be collected (added) and
    how they should be finalized (packed) combined with optional historical snapshots.
    """

    @abc.abstractmethod
    def add_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Accumulates a batch of hidden states into the aggregator.

        Args:
            hidden_states: The tensor containing the hidden states to process.
        """

    @abc.abstractmethod
    def pack_with_snapshot(self, snapshot: torch.Tensor | None) -> torch.Tensor | None:
        """Finalizes the aggregation and combines it with an optional previous snapshot.

        This method typically retrieves the accumulated states, processes them
        (if not done during addition), and concatenates them with the snapshot.

        Args:
            snapshot: An optional tensor representing previously aggregated states
                to be prepended to the current collection.

        Returns:
            The combined result of the snapshot and the newly aggregated states,
            or None if no states were collected.
        """
