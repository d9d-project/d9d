from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

from d9d.metric.component import MetricAccumulator

from .confusion_matrix import ConfusionMatrix


class ConfusionMatrixAccumulator(Stateful):
    """Accumulates confusion matrix statistics across batches and distributed workers."""

    def __init__(self, num_outputs: int):
        """Constructs the ``ConfusionMatrixAccumulator`` object.

        Args:
            num_outputs: The number of distinct classes to track.
        """
        self._num_outputs = num_outputs
        self._tp = MetricAccumulator(torch.zeros(num_outputs, dtype=torch.long))
        self._fp = MetricAccumulator(torch.zeros(num_outputs, dtype=torch.long))
        self._tn = MetricAccumulator(torch.zeros(num_outputs, dtype=torch.long))
        self._fn = MetricAccumulator(torch.zeros(num_outputs, dtype=torch.long))

    @property
    def state(self) -> ConfusionMatrix:
        """The accumulated confusion matrix, with counts of shape ``(num_outputs,)``."""
        return ConfusionMatrix(tp=self._tp.value, fp=self._fp.value, tn=self._tn.value, fn=self._fn.value)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Updates the accumulated statistics with a new batch of predictions and targets.

        Args:
            preds: Binary predictions, as returned by a ``ClassificationPredictionsProcessor``.
                Shape: ``(..., num_outputs)``.
            targets: Binary targets, as returned by a ``ClassificationPredictionsProcessor``.
                Shape: ``(..., num_outputs)``.

        Raises:
            ValueError: If ``preds`` and ``targets`` have different shapes, or if their last dimension does not
                match ``num_outputs``.
        """
        if preds.shape != targets.shape:
            raise ValueError(f"preds shape ({tuple(preds.shape)}) must match targets shape ({tuple(targets.shape)}).")

        if preds.shape[-1] != self._num_outputs:
            raise ValueError(
                f"The last dimension of preds ({preds.shape[-1]}) must equal num_outputs ({self._num_outputs})."
            )

        preds = preds.long().flatten(0, -2)
        targets = targets.long().flatten(0, -2)

        # Sum over all leading dimensions, flattened into dim 0; result shape: (num_outputs,).
        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1 - targets)).sum(dim=0)
        fn = ((1 - preds) * targets).sum(dim=0)
        tn = ((1 - preds) * (1 - targets)).sum(dim=0)

        self._tp.update(tp)
        self._fp.update(fp)
        self._tn.update(tn)
        self._fn.update(fn)

    def sync(self):
        """Synchronizes the accumulated counts across the default process group."""
        self._tp.sync()
        self._fp.sync()
        self._tn.sync()
        self._fn.sync()

    def reset(self):
        """Resets all internal metric accumulators to zero."""
        self._tp.reset()
        self._fp.reset()
        self._tn.reset()
        self._fn.reset()

    def to(self, device: str | torch.device | int):
        """Moves the internal accumulators to a device.

        Args:
            device: The target device.
        """
        self._tp.to(device)
        self._fp.to(device)
        self._tn.to(device)
        self._fn.to(device)

    def state_dict(self) -> dict[str, Any]:
        """Returns the state dictionary of the accumulator.

        Returns:
            The states of all internal accumulators.
        """
        return {
            "tp": self._tp.state_dict(),
            "fp": self._fp.state_dict(),
            "tn": self._tn.state_dict(),
            "fn": self._fn.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restores the accumulator state from the given state dictionary.

        Args:
            state_dict: The state dictionary to load.
        """
        self._tp.load_state_dict(state_dict["tp"])
        self._fp.load_state_dict(state_dict["fp"])
        self._tn.load_state_dict(state_dict["tn"])
        self._fn.load_state_dict(state_dict["fn"])
