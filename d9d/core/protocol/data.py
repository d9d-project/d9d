from typing import Protocol, runtime_checkable

from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.types import Microbatches


@runtime_checkable
class BatchIterator(Stateful, Protocol):
    """A stateful iterator over packs that the train/eval loop drives.

    A ``BatchIterator`` yields packs - one pack is exactly one step's worth of microbatches, ready
    for the execution engine. It is the single checkpoint boundary for the data stream: it saves and
    restores its own position so resumption is exact. It may optionally also be ``Sized``.

    A ``BatchIterator`` yields CPU (optionally memory-pinned) tensors; moving each pack to the device
    is the loop's responsibility.
    """

    def __iter__(self) -> "BatchIterator":
        """Returns the iterator itself.

        Returns:
            This iterator.
        """
        ...

    def __next__(self) -> Microbatches:
        """Advances to the next pack.

        Returns:
            The next pack (one step's worth of microbatches).

        Raises:
            StopIteration: If the underlying data stream is exhausted.
        """
        ...
