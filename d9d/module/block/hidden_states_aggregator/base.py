import abc

import torch


class BaseHiddenStatesAggregator(abc.ABC):
    """Abstract base class for hidden states aggregation strategies.

    An aggregator collects hidden states and then packs them together with an optional earlier snapshot.
    """

    @abc.abstractmethod
    def add_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Accumulates a batch of hidden states into the aggregator.

        Args:
            hidden_states: Hidden states to aggregate. Shape: ``(batch, seq_len, hidden_size)``.
        """

    @abc.abstractmethod
    def pack_with_snapshot(self, snapshot: torch.Tensor | None) -> torch.Tensor | None:
        """Finalizes the aggregation and combines it with an optional earlier snapshot.

        Args:
            snapshot: Previously aggregated states to prepend to the current collection, or ``None``.

        Returns:
            The snapshot followed by the newly aggregated states, or ``None`` if no states were collected.
        """
