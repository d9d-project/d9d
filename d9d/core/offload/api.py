import dataclasses
from enum import StrEnum
from typing import Protocol, runtime_checkable

from d9d.core.dist_context import DistributedContext


class SleepTag(StrEnum):
    """Subsystems that ``Trainer.sleep`` and ``Trainer.wake`` act on.

    Attributes:
        TENSOR_STATES: All GPU tensor state: model parameters and buffers, optimizer state, gradient buckets and
            the residual loss accumulator. They are always offloaded together.
        COMMS: NCCL process groups. Opt-in. It is not implemented yet, so requesting it raises
            ``NotImplementedError``.
    """

    TENSOR_STATES = "tensor_states"
    COMMS = "comms"


DEFAULT_SLEEP_TAGS = frozenset({SleepTag.TENSOR_STATES})
"""The default tags for ``Trainer.sleep`` and ``Trainer.wake``: tensor state only, no comms."""


@dataclasses.dataclass(kw_only=True, frozen=True)
class OffloadContext:
    """Context passed to ``Offloadable.offload``.

    Attributes:
        dist_context: The distributed context the subsystem was built under.
        pin_memory: Whether to allocate the host buffer in pinned memory.
    """

    dist_context: DistributedContext
    pin_memory: bool


@dataclasses.dataclass(kw_only=True, frozen=True)
class OnloadContext:
    """Context passed to ``Offloadable.onload``.

    Attributes:
        dist_context: The distributed context the subsystem was built under.
    """

    dist_context: DistributedContext


@runtime_checkable
class Offloadable(Protocol):
    """Protocol for subsystems that own GPU-resident state and can release it to host memory.

    An ``offload`` followed by an ``onload`` must not change anything observable. Parameter identities,
    optimizer state keys, ``DTensor`` instances, placements and dtypes stay the same. Only the device storage
    is allocated again.
    """

    def offload(self, ctx: OffloadContext) -> None:
        """Releases the GPU memory owned by this subsystem, moving its state to host memory.

        Args:
            ctx: Context for this operation.
        """

    def onload(self, ctx: OnloadContext) -> None:
        """Moves the state that ``offload`` released back to GPU memory.

        Args:
            ctx: Context for this operation.
        """

    def is_offloaded(self) -> bool:
        """Reports whether this subsystem currently has its state in host memory.

        Returns:
            ``True`` if the subsystem is offloaded, ``False`` otherwise.
        """
