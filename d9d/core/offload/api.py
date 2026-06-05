import dataclasses
from enum import StrEnum
from typing import Protocol, runtime_checkable

from d9d.core.dist_context import DistributedContext


class SleepTag(StrEnum):
    """Subsystem selector for Trainer.sleep and Trainer.wake.

    Attributes:
        TENSOR_STATES: All GPU tensor state - model parameters and buffers, optimizer state,
            gradient buckets and the residual loss accumulator. Always offloaded as a unit.
        COMMS: NCCL process groups. Opt-in; its implementation is deferred to a second phase,
            so requesting it currently raises NotImplementedError.
    """

    TENSOR_STATES = "tensor_states"
    COMMS = "comms"


DEFAULT_SLEEP_TAGS = frozenset({SleepTag.TENSOR_STATES})
"""The default tag set for Trainer.sleep and Trainer.wake: tensor state only, no comms."""


@dataclasses.dataclass(kw_only=True, frozen=True)
class OffloadContext:
    """Context passed to Offloadable.offload.

    Attributes:
        dist_context: The distributed context the subsystem was built under.
        pin_memory: Whether to allocate the host buffer in pinned memory.
    """

    dist_context: DistributedContext
    pin_memory: bool


@dataclasses.dataclass(kw_only=True, frozen=True)
class OnloadContext:
    """Context passed to Offloadable.onload.

    Attributes:
        dist_context: The distributed context the subsystem was built under.
    """

    dist_context: DistributedContext


@runtime_checkable
class Offloadable(Protocol):
    """Protocol for subsystems that own GPU-resident state and can release it to host memory.

    An "offload" followed by an "onload" must be observationally a no-op: parameter identities,
    optimizer state keys, DTensor wrapper instances, placements and dtypes are all preserved
    across the round trip. Only the underlying device storages are reallocated.
    """

    def offload(self, ctx: OffloadContext) -> None:
        """Releases the GPU memory owned by this subsystem, moving its state to host memory.

        Args:
            ctx: Context for this operation.
        """

    def onload(self, ctx: OnloadContext) -> None:
        """Restores GPU residency of the state previously released by "offload".

        Args:
            ctx: Context for this operation.
        """

    def is_offloaded(self) -> bool:
        """Reports whether this subsystem currently has its state on host memory.

        Returns:
            True if the subsystem is offloaded, False otherwise.
        """
