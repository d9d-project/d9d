import pickle  # noqa: S403
import random
from typing import Any, Protocol, TypeVar

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset

TDatasetReturn_co = TypeVar("TDatasetReturn_co", covariant=True)


class DatasetImplementingSortKeyProtocol(Protocol[TDatasetReturn_co]):
    """
    Protocol for datasets that support retrieval of a specific key for sorting purposes.

    This is typically used for length-based bucketing/sorting where the dataset
    needs to expose the length of an item without loading the full item.
    """

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        ...

    def sort_key(self, index: int) -> Any:
        """
        Returns a value used for sorting the dataset at the given index.

        Args:
            index: The index of the item.

        Returns:
            A comparable value (e.g., int length) used for sorting.
        """
        ...

    def __getitem__(self, item: int) -> TDatasetReturn_co:
        """Retrieves the item at the specific index."""
        ...


class BufferSortedDataset(Dataset, Stateful):
    """
    A dataset wrapper that groups items into buffers, sorts them, and yields them with local shuffling.

    This prevents extreme padding in variable-length training (by grouping similar lengths)
    while maintaining enough randomness to ensure statistical variance in updates.

    Algorithm:

    1. Select a range of indices (size `buffer_size`).
    2. Sort these indices based on `base_dataset.sort_key()`.
    3. Break the sorted list into packs of size `pack_size`.
    4. Shuffle the order of these packs.
    5. Flatten the list and serve items.
    """

    def __init__(
            self,
            base_dataset: DatasetImplementingSortKeyProtocol[TDatasetReturn_co],
            buffer_size: int,
            pack_size: int,
            init_seed: int | None = None
    ):
        """
        Constructs a BufferSortedDataset object.

        Args:
            base_dataset: The underlying dataset implementing the `DatasetImplementingSortKeyProtocol` protocol.
            buffer_size: The number of items to load into the buffer for sorting.
            pack_size: The size of local groups (batches/micro-batches) that remain
                contiguous after sorting, but are shuffled relative to other packs.
            init_seed: Seed for the random number generator used for shuffling packs.
        """

        self._base_dataset = base_dataset
        self._buffer_size = buffer_size
        self._pack_size = pack_size

        self._rng = random.Random(init_seed ^ 0x105E7)
        self._buffer_indices: list[int] = []
        self._buffer_idx: int = -1

    def _update_buffer_idx(self, buffer_idx: int):
        select_start = buffer_idx * self._buffer_size
        select_end = (buffer_idx + 1) * self._buffer_size
        select_end = min(select_end, len(self._base_dataset))

        base_idx = list(range(select_start, select_end))
        base_sort_keys = [self._base_dataset.sort_key(idx) for idx in range(select_start, select_end)]

        local_idx = list(range(len(base_idx)))
        local_idx = sorted(local_idx, key=lambda local_id: base_sort_keys[local_id])

        local_idx = [
            local_idx[i: i + self._pack_size]
            for i in range(0, len(local_idx), self._pack_size)
        ]
        self._rng.shuffle(local_idx)
        local_idx = [y for x in local_idx for y in x]

        self._buffer_indices = [base_idx[local_id] for local_id in local_idx]

        self._buffer_idx = buffer_idx

    def __getitem__(self, index: int) -> TDatasetReturn_co:
        """
        Retrieves an item from the locally sorted/shuffled buffer.

        Args:
            index: The global index.

        Returns:
            The dataset item.
        """

        needs_buffer_idx = index // self._buffer_size
        if self._buffer_idx != needs_buffer_idx:
            self._update_buffer_idx(needs_buffer_idx)

        take_id = self._buffer_indices[index % self._buffer_size]

        return self._base_dataset[take_id]

    def __len__(self):
        """Returns the length of the base dataset."""

        return len(self._base_dataset)

    def state_dict(self) -> dict[str, Any]:
        ret = {
            "seed": pickle.dumps(self._rng.getstate()),
            "buffer_idx": self._buffer_idx,
            "buffer_indices": self._buffer_indices,
        }
        if isinstance(self._base_dataset, Stateful):
            ret["base_dataset"] = self._base_dataset.state_dict()
        return ret

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._rng.setstate(pickle.loads(state_dict["seed"]))  # noqa: S301
        self._buffer_idx = state_dict["buffer_idx"]
        self._buffer_indices = state_dict["buffer_indices"]
        if isinstance(self._base_dataset, Stateful):
            self._base_dataset.load_state_dict(state_dict["base_dataset"])
