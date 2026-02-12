import torch

from .base import BaseHiddenStatesAggregator


def _aggregate_hidden_states(hidden_states: torch.Tensor, agg_mask: torch.Tensor) -> torch.Tensor:
    orig_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    num_tokens = agg_mask.sum(dim=1)[:, None]
    masked_states = hidden_states * agg_mask[:, :, None]
    averaged_states = masked_states.sum(dim=1) / num_tokens
    return averaged_states.to(orig_dtype)


class HiddenStatesAggregatorMean(BaseHiddenStatesAggregator):
    """Aggregator that computes the mean of hidden states using a validity mask."""

    def __init__(self, agg_mask: torch.Tensor) -> None:
        """Constructs the mean aggregator with the given mask.

        Args:
            agg_mask: A tensor used to mask out padding or invalid tokens
                during average calculation.
        """
        self._agg_mask = agg_mask
        self._collected_states: list[torch.Tensor] = []

    def add_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Calculates the masked mean immediately and stores the result.

        Args:
            hidden_states: The raw hidden states to be averaged and stored.
        """
        agg = _aggregate_hidden_states(hidden_states=hidden_states, agg_mask=self._agg_mask)
        self._collected_states.append(agg)

    def pack_with_snapshot(self, snapshot: torch.Tensor | None) -> torch.Tensor | None:
        """Stacks collected projected averages and appends to the snapshot.

        This operation clears the internal buffer of collected states.

        Args:
            snapshot: Previous states to prepend.

        Returns:
            A tensor containing the snapshot followed by the stacked collected states,
            or None if nothing was collected.
        """
        if len(self._collected_states) == 0:
            return None

        stacked = torch.stack(self._collected_states, dim=0)
        self._collected_states.clear()
        if snapshot is not None:
            stacked = torch.cat([snapshot, stacked], dim=0)
        return stacked
