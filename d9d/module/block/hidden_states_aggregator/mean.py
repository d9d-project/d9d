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
    """Aggregator that computes the masked mean of hidden states over the sequence."""

    def __init__(self, agg_mask: torch.Tensor) -> None:
        """Constructs the ``HiddenStatesAggregatorMean`` object.

        Args:
            agg_mask: Mask of the tokens to include in the mean. Zeros mask out padding or invalid tokens.
                Shape: ``(batch, seq_len)``.
        """
        self._agg_mask = agg_mask
        self._collected_states: list[torch.Tensor] = []

    def add_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Computes the masked mean of the hidden states and stores it.

        Args:
            hidden_states: Hidden states to average. Shape: ``(batch, seq_len, hidden_size)``.
        """
        agg = _aggregate_hidden_states(hidden_states=hidden_states, agg_mask=self._agg_mask)
        self._collected_states.append(agg)

    def pack_with_snapshot(self, snapshot: torch.Tensor | None) -> torch.Tensor | None:
        """Stacks the collected means after the snapshot and clears the collected means.

        Args:
            snapshot: Previous means to prepend, or ``None``. Shape: ``(num_snapshot_means, batch, hidden_size)``.

        Returns:
            The snapshot followed by the stacked collected means, or ``None`` if nothing was collected.
            Shape: ``(num_means, batch, hidden_size)``.
        """
        if len(self._collected_states) == 0:
            return None

        stacked = torch.stack(self._collected_states, dim=0)
        self._collected_states.clear()
        if snapshot is not None:
            stacked = torch.cat([snapshot, stacked], dim=0)
        return stacked
