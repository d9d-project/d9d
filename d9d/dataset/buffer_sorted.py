import pickle  # noqa: S403 - only for the RNG state that this class writes and reads itself
import random
from typing import Any, Protocol, TypeVar

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset

_T_co = TypeVar("_T_co", covariant=True)


class DatasetImplementingSortKeyProtocol(Protocol[_T_co]):
    """Protocol for datasets that return a sort key for an item without loading it.

    It is typically used for length-based bucketing, where the dataset exposes the length of an item.
    """

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        ...

    def sort_key(self, index: int) -> Any:
        """Returns a value used for sorting the dataset at the given index.

        Args:
            index: The index of the item.

        Returns:
            A comparable value used for sorting, e.g. the item length.
        """
        ...

    def __getitem__(self, item: int) -> _T_co:
        """Returns the item at the given index."""
        ...


class BufferSortedDataset(Dataset[_T_co], Stateful):
    """Wraps a dataset to serve items sorted within buffers, with local shuffling.

    Items of similar length end up in the same pack, which reduces padding in variable-length training.
    The shuffling keeps enough randomness in the order of updates.

    Algorithm:

    1.  Select a range of ``buffer_size`` indices.
    2.  Build sort keys: ``(base_dataset.sort_key(index), random_tie_breaker)``.
    3.  Sort the indices by these keys.
    4.  Split the sorted list into packs of ``pack_size`` items.
    5.  Shuffle the order of the packs.
    6.  Shuffle the items within each pack.
    7.  Flatten and serve.
    """

    def __init__(
        self,
        base_dataset: DatasetImplementingSortKeyProtocol[_T_co],
        buffer_size: int,
        pack_size: int,
        init_seed: int | None = None,
    ):
        """Constructs the ``BufferSortedDataset`` object.

        Args:
            base_dataset: The underlying dataset.
            buffer_size: The number of items sorted together.
            pack_size: The size of a group of similar items, usually the batch or microbatch size.
            init_seed: Seed for the random number generator.
        """
        self._base_dataset = base_dataset
        self._buffer_size = buffer_size
        self._pack_size = pack_size

        self._rng = random.Random(init_seed ^ 0x105E7 if init_seed is not None else None)
        self._buffer_indices: list[int] = []
        self._buffer_idx: int = -1

    def _update_buffer_idx(self, buffer_idx: int):
        select_start = buffer_idx * self._buffer_size
        select_end = (buffer_idx + 1) * self._buffer_size
        select_end = min(select_end, len(self._base_dataset))

        base_idx = list(range(select_start, select_end))

        sort_keys = [
            # The random tie-breaker shuffles items with equal sort keys.
            (self._base_dataset.sort_key(idx), self._rng.random())
            for idx in base_idx
        ]

        local_idx = list(range(len(base_idx)))
        local_idx.sort(key=lambda i: sort_keys[i])

        local_idx_packs = [local_idx[i : i + self._pack_size] for i in range(0, len(local_idx), self._pack_size)]

        self._rng.shuffle(local_idx_packs)

        for pack in local_idx_packs:
            self._rng.shuffle(pack)

        flat_local_idx = [y for x in local_idx_packs for y in x]

        self._buffer_indices = [base_idx[local_id] for local_id in flat_local_idx]
        self._buffer_idx = buffer_idx

    def __getitem__(self, index: int) -> _T_co:
        needs_buffer_idx = index // self._buffer_size
        if self._buffer_idx != needs_buffer_idx:
            self._update_buffer_idx(needs_buffer_idx)

        take_id = self._buffer_indices[index % self._buffer_size]
        return self._base_dataset[take_id]

    def __len__(self) -> int:
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
        self._rng.setstate(pickle.loads(state_dict["seed"]))  # noqa: S301 - our own checkpoint, from state_dict()
        self._buffer_idx = state_dict["buffer_idx"]
        self._buffer_indices = state_dict["buffer_indices"]
        if isinstance(self._base_dataset, Stateful):
            self._base_dataset.load_state_dict(state_dict["base_dataset"])
