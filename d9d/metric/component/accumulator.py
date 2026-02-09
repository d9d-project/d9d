from enum import StrEnum
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful


# we explicitly do not add 'avg' op since it is not safe for metric accumulation
class MetricReduceOp(StrEnum):
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
            raise ValueError("Unknown metric reduce op")


def _accumulate_inplace_(op: MetricReduceOp, accumulator: torch.Tensor, value: torch.Tensor):
    match op:
        case MetricReduceOp.sum:
            accumulator.add_(value)
        case MetricReduceOp.max:
            accumulator.copy_(torch.maximum(accumulator, value))
        case MetricReduceOp.min:
            accumulator.copy_(torch.minimum(accumulator, value))


class MetricAccumulator(Stateful):
    """Helper class to track a distributed metric state.

    This class manages two copies of the state: a 'local' copy that is updated
    locally on every step, and a 'synchronized' copy that is populated during
    the sync phase via distributed reduction (all-reduce).
    """

    def __init__(
            self,
            initial_value: torch.Tensor,
            reduce_op: MetricReduceOp = MetricReduceOp.sum
    ):
        """Constructs MetricAccumulator object.

        Args:
            initial_value: Tensor representing the starting value (e.g., 0 for sum, -inf for max).
                This tensor determines the device and dtype of the accumulator.
            reduce_op: The reduction operation to use during updates and synchronization.
        """

        self._initial = initial_value.clone()

        self._local = initial_value.clone()
        self._synchronized = initial_value.clone()

        self._reduce_op = reduce_op

        self._is_synchronized = False

    def update(self, value: torch.Tensor):
        """Updates the local accumulator with a new value.

        This operation is performed in-place on the local tensor using the
        configured reduction operation (e.g., add for Sum, max for Max).
        It marks the accumulator as not synchronized.

        Args:
            value: The value to accumulate.
        """

        _accumulate_inplace_(self._reduce_op, self._local, value)

        self._is_synchronized = False

    def sync(self):
        """Synchronizes the accumulator across the default distributed process group.

        This method acts as a blocking barrier. It copies the local state to a buffer
        and performs an `all_reduce` collective operation.
        """

        self._synchronized.copy_(self._local)
        dist.all_reduce(self._synchronized, op=_torch_reduce_op_for(self._reduce_op))

        self._is_synchronized = True

    @property
    def value(self) -> torch.Tensor:
        """Returns the current accumulated value.

        Returns:
            The global synchronized value if `sync()` was called recently,
            otherwise the local accumulated value.
        """

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

        return {
            "local": self._local,
            "synchronized": self._synchronized,
            "is_synchronized": self._is_synchronized
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the accumulator state from a checkpoint.

        Args:
            state_dict: Dictionary containing state to load.
        """

        self._local = state_dict["local"]
        self._synchronized = state_dict["synchronized"]
        self._is_synchronized = state_dict["is_synchronized"]
