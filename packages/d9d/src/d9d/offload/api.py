import dataclasses
from typing import Protocol, runtime_checkable

from d9d.distributed import DistributedContext


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
