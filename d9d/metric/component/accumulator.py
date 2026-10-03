from enum import StrEnum
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful


# No "avg" op: averaging running values or per-rank averages gives wrong results when counts differ.
class MetricReduceOp(StrEnum):
    """Reduction operation of a ``MetricAccumulator``, used both for updates and for synchronization.

    Attributes:
        sum: Adds values.
        max: Keeps the maximum value.
        min: Keeps the minimum value.
    """

    sum = "sum"
    max = "max"
    min = "min"


def _torch_reduce_op_for(op: MetricReduceOp) -> dist.ReduceOp.RedOpType:
    match op:
        case MetricReduceOp.sum:
            return dist.ReduceOp.SUM
        case MetricReduceOp.max:
            return dist.ReduceOp.MAX
        case MetricReduceOp.min:
            return dist.ReduceOp.MIN
        case _:
            raise ValueError(f"Unknown metric reduce op ({op}).")


def _accumulate_inplace_(op: MetricReduceOp, accumulator: torch.Tensor, value: torch.Tensor | float | bool):
    match op:
        case MetricReduceOp.sum:
            accumulator.add_(value)
        case MetricReduceOp.max:
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Value type ({type(value).__name__}) is not supported by the max reduce op. Pass a tensor."
                )
            accumulator.copy_(torch.maximum(accumulator, value))
        case MetricReduceOp.min:
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Value type ({type(value).__name__}) is not supported by the min reduce op. Pass a tensor."
                )
            accumulator.copy_(torch.minimum(accumulator, value))


class MetricAccumulator(Stateful):
    """Accumulator of a distributed metric state.

    It keeps two copies of the state: a local copy, updated on every step, and a synchronized copy, filled by an
    all-reduce in ``sync()``.
    """

    def __init__(self, initial_value: torch.Tensor, reduce_op: MetricReduceOp = MetricReduceOp.sum):
        """Constructs the ``MetricAccumulator`` object.

        Args:
            initial_value: The starting value, e.g. 0 for sum or -inf for max. It sets the device and dtype of the
                accumulator.
            reduce_op: The reduction operation for updates and synchronization.
        """
        self._initial = initial_value.clone()

        self._local = initial_value.clone()
        self._synchronized = initial_value.clone()

        self._reduce_op = reduce_op

        self._is_synchronized = False

    def update(self, value: torch.Tensor | float | bool):
        """Accumulates a value into the local state with the configured reduction operation.

        After an update, ``value`` returns the local state until the next ``sync()``.

        Args:
            value: The value to accumulate. The ``max`` and ``min`` operations accept only tensors.

        Raises:
            ValueError: If ``value`` is not a tensor and the reduction operation is ``max`` or ``min``.
        """
        _accumulate_inplace_(self._reduce_op, self._local, value)

        self._is_synchronized = False

    def sync(self):
        """Synchronizes the accumulator across the default process group.

        Every rank must call it. The local state stays unchanged.
        """
        self._synchronized.copy_(self._local)
        dist.all_reduce(self._synchronized, op=_torch_reduce_op_for(self._reduce_op))

        self._is_synchronized = True

    @property
    def value(self) -> torch.Tensor:
        """The synchronized value if ``sync()`` was called after the last update, otherwise the local value."""
        return self._synchronized if self._is_synchronized else self._local

    def reset(self):
        """Resets the accumulator to its initial state."""
        self._local.copy_(self._initial)

        self._is_synchronized = False

    def to(self, device: str | torch.device | int):
        """Moves internal tensors to the specified device.

        Args:
            device: Target device.
        """
        self._local = self._local.to(device)
        self._synchronized = self._synchronized.to(device)

    def state_dict(self) -> dict[str, Any]:
        """Returns the serialized state of the accumulator.

        Returns:
            Dictionary containing local and synchronized tensors and status flags.
        """
        return {"local": self._local, "synchronized": self._synchronized, "is_synchronized": self._is_synchronized}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the accumulator state from a checkpoint.

        Args:
            state_dict: Dictionary containing state to load.
        """
        self._local = state_dict["local"]
        self._synchronized = state_dict["synchronized"]
        self._is_synchronized = state_dict["is_synchronized"]
