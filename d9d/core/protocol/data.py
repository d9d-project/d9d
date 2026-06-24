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
class MicrobatchPackIterator(Stateful, Protocol):
    """Protocol defining an interface for a stateful iterator over microbatch packs that the loop drives.

    This protocol ensures that the iterator yields packs - one pack is exactly one step's worth of
    microbatches - and supports state checkpointing via the Stateful interface, acting as the single
    checkpoint boundary for the data stream. It may optionally also be Sized, reporting the number of
    steps it will yield. It yields CPU (optionally memory-pinned) tensors; moving each pack to the
    device is the loop's responsibility.
    """

    def __iter__(self) -> "MicrobatchPackIterator":
        """Returns the iterator itself.

        Returns:
            This iterator.
        """
        ...

    def __next__(self) -> MicrobatchPack:
        """Advances to the next pack.

        Returns:
            The next microbatch pack (one step's worth of microbatches).

        Raises:
            StopIteration: If the underlying data stream is exhausted.
        """
        ...
