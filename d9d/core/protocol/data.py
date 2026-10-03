from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from d9d.core.types import MicrobatchPack, PyTree


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol for a sized, stateful stream of single microbatches.

    A conforming loader yields one collated microbatch at a time and reports its length in microbatches.
    It supports checkpointing through the ``Stateful`` interface (``state_dict`` and ``load_state_dict``).

    A torchdata ``StatefulDataLoader`` conforms to it.
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

    def state_dict(self) -> dict[str, Any]:
        """Returns the loader's checkpointable state.

        Returns:
            A dictionary representing the loader's state.
        """
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the loader's state from a previously produced state dict.

        Args:
            state_dict: The state dict to restore from.
        """
        ...


@runtime_checkable
class MicrobatchPackStream(Protocol):
    """Protocol for a stateful stream of microbatch packs that the loop drives.

    One pack holds exactly one step's worth of microbatches. The stream supports checkpointing through the
    ``Stateful`` interface (``state_dict`` and ``load_state_dict``). It is the only checkpoint boundary of the
    data pipeline.

    The stream yields CPU tensors, which can be in pinned memory. The loop moves each pack to the device.
    When prefetching, the loop iterates the stream on a background thread.
    """

    def __iter__(self) -> Iterator[MicrobatchPack]:
        """Returns an iterator over microbatch packs.

        Returns:
            An iterator yielding one microbatch pack (one step's worth of microbatches) at a time.
        """
        ...

    @property
    def total_steps(self) -> int | None:
        """The number of steps (packs) this stream yields, or ``None`` if it is not known ahead of time.

        A streaming or data-dependent source returns ``None``. In that case, the job duration must come from
        ``JobScheduleConfig``.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Returns the stream's checkpointable state.

        When prefetching, the loop calls this method after every pack. Keep it cheap, and do not return
        objects that later iteration mutates.

        Returns:
            A dictionary representing the stream's state.
        """
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the stream's state from a previously produced state dict.

        Args:
            state_dict: The state dict to restore from.
        """
        ...
