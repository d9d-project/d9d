from collections.abc import Sequence

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

    def collect(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Computes the masked mean of the hidden states.

        Args:
            hidden_states: Hidden states to average. Shape: ``(batch, seq_len, hidden_size)``.

        Returns:
            The mean. Shape: ``(batch, hidden_size)``.
        """
        return (_aggregate_hidden_states(hidden_states=hidden_states, agg_mask=self._agg_mask),)

    def pack(self, collected: Sequence[torch.Tensor], snapshot: torch.Tensor | None) -> torch.Tensor | None:
        """Stacks the collected means after the snapshot.

        Args:
            collected: The collected means, in order.
            snapshot: Previous means to prepend, or ``None``. Shape: ``(num_snapshot_means, batch, hidden_size)``.

        Returns:
            The snapshot followed by the stacked collected means, or only the snapshot if nothing was collected.
            Shape: ``(num_means, batch, hidden_size)``.
        """
        if len(collected) == 0:
            return snapshot

        stacked = torch.stack(tuple(collected), dim=0)
        if snapshot is not None:
            stacked = torch.cat([snapshot, stacked], dim=0)
        return stacked
