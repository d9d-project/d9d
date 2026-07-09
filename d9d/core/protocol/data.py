from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.types import MicrobatchPack, PyTree


@runtime_checkable
class DataLoaderProtocol(Stateful, Protocol):
    """Protocol defining an interface for a sized, stateful stream of single microbatches.

    This protocol ensures that the loader yields one collated microbatch at a time, reports its
    length in microbatches, and supports state checkpointing via the Stateful interface.

    A torchdata ``StatefulDataLoader`` satisfies it out of the box.
    """

    def __iter__(self) -> Iterator[PyTree]:
        """Returns an iterator over single collated microbatches.

        Returns:
            An iterator yielding one collated microbatch at a time.
        """
        ...

    def __len__(self) -> int:
        """Returns the number of microbatches this loader yields.

        Returns:
            The number of microbatches.
        """
        ...


@runtime_checkable
class MicrobatchPackStream(Stateful, Protocol):
    """Protocol defining an interface for a stateful, iterable stream of microbatch packs that the loop drives.

    This protocol ensures that iterating the stream yields packs - one pack is exactly one step's
    worth of microbatches - and that it supports state checkpointing via the Stateful interface,
    acting as the single checkpoint boundary for the data stream. It yields CPU (optionally
    memory-pinned) tensors; moving each pack to the device is the loop's responsibility.
    """

    def __iter__(self) -> Iterator[MicrobatchPack]:
        """Returns an iterator over microbatch packs.

        Returns:
            An iterator yielding one microbatch pack (one step's worth of microbatches) at a time.
        """
        ...

    @property
    def total_steps(self) -> int | None:
        """The number of steps (packs) this stream will yield, if known.

        Returns:
            The step count, or ``None`` when it cannot be determined ahead of time (e.g. a streaming
            or data-dependent source). When ``None``, the job duration must come from ``JobScheduleConfig``.
        """
        ...
