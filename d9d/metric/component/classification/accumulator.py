from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

from d9d.metric.component import MetricAccumulator

from .confusion_matrix import ConfusionMatrix


class ConfusionMatrixAccumulator(Stateful):
    """Accumulates confusion matrix statistics across batches and distributed workers."""

    def __init__(self, num_outputs: int):
        """Constructs the ConfusionMatrixAccumulator object.

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
        """Provides the current accumulated state.

        Returns:
            A single confusion matrix containing 1D tensors of counts for each
            tracked output/class.
        """

        return ConfusionMatrix(tp=self._tp.value, fp=self._fp.value, tn=self._tn.value, fn=self._fn.value)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Updates the accumulated statistics with a new batch of predictions and targets.

        Args:
            preds: Pre-processed binary predictions tensor.
            targets: Pre-processed binary targets tensor.

        Raises:
            ValueError: If predictions and targets have mismatched shapes, or if the
                last dimension does not match the configured number of outputs.
        """

        # preds/targets are pre-processed binary tensors
        if preds.shape != targets.shape:
            raise ValueError(f"preds and targets must have same shape, got {preds.shape} and {targets.shape}")

        if preds.shape[-1] != self._num_outputs:
            raise ValueError(f"Expected {self._num_outputs} outputs, got {preds.shape[1]}")

        preds = preds.long().flatten(0, -2)
        targets = targets.long().flatten(0, -2)

        # calculation over batch dimension (dim 0)
        # Result shape: (num_outputs,)
        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1 - targets)).sum(dim=0)
        fn = ((1 - preds) * targets).sum(dim=0)
        tn = ((1 - preds) * (1 - targets)).sum(dim=0)

        self._tp.update(tp)
        self._fp.update(fp)
        self._tn.update(tn)
        self._fn.update(fn)

    def sync(self):
        """Synchronizes the accumulated metrics across all distributed workers."""

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
        """Moves the underlying metric accumulators to the specified target device.

        Args:
            device: The target device to move the internal tensors to.
        """

        self._tp.to(device)
        self._fp.to(device)
        self._tn.to(device)
        self._fn.to(device)

    def state_dict(self) -> dict[str, Any]:
        """Retrieves the current state dictionary of the accumulator.

        Returns:
            A dictionary containing the state bounds of all internal accumulators.
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
            state_dict: The state dictionary to inject into the accumulator.
        """

        self._tp.load_state_dict(state_dict["tp"])
        self._fp.load_state_dict(state_dict["fp"])
        self._tn.load_state_dict(state_dict["tn"])
        self._fn.load_state_dict(state_dict["fn"])
