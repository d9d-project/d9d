import abc
from collections.abc import Sequence

import torch


class BaseHiddenStatesAggregator(abc.ABC):
    """Abstract base class for hidden states aggregation strategies.

    An aggregator reduces hidden states and then packs them together with an optional earlier snapshot. It
    keeps no state: the collected values stay with the caller, so they can pass through a chain of segments.
    """

    @abc.abstractmethod
    def collect(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Reduces a batch of hidden states.

        Args:
            hidden_states: Hidden states to aggregate. Shape: ``(batch, seq_len, hidden_size)``.

        Returns:
            The values to keep until ``pack``. Empty if the aggregator keeps nothing.
        """

    @abc.abstractmethod
    def pack(self, collected: Sequence[torch.Tensor], snapshot: torch.Tensor | None) -> torch.Tensor | None:
        """Finalizes the aggregation and combines it with an optional earlier snapshot.

        Args:
            collected: The values that ``collect`` returned, in order.
            snapshot: Previously aggregated states to prepend to ``collected``, or ``None``.

        Returns:
            The snapshot followed by the newly aggregated states, or only the snapshot if no states were collected.
        """
