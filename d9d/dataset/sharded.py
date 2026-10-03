import math
from collections.abc import Sized
from enum import StrEnum
from typing import Any, TypeVar

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset

from d9d.core.dist_context import BATCH_DOMAIN, DistributedContext


class ShardIndexingMode(StrEnum):
    """Defines how a dataset is split across shards.

    The examples show 14 items split across 4 shards.

    Attributes:
        sequential: Round-robin distribution.

            shard0: 0, 4, 8, 12
            shard1: 1, 5, 9, 13
            shard2: 2, 6, 10
            shard3: 3, 7, 11

        chunked: Contiguous blocks.

            shard0: 0, 1, 2, 3
            shard1: 4, 5, 6, 7
            shard2: 8, 9, 10, 11
            shard3: 12, 13
    """

    sequential = "sequential"
    chunked = "chunked"


_T_co = TypeVar("_T_co", covariant=True)


class ShardedDataset(Dataset[_T_co], Stateful):
    """Wraps a dataset to expose only one of its shards.

    Use it for data-parallel training, where each process sees only a subset of the data. Optional padding gives
    all shards the same length, so no rank waits forever in a collective.
    """

    def __init__(
        self,
        dataset: Dataset[_T_co],
        total_shards: int,
        current_shard: int,
        indexing_mode: ShardIndexingMode,
        pad_to_equal_size_across_shards: bool,
    ):
        """Constructs the ``ShardedDataset`` object.

        Args:
            dataset: The underlying dataset to shard.
            total_shards: The total number of shards, e.g. the number of data-parallel ranks.
            current_shard: The index of the current shard, e.g. the current data-parallel rank.
            indexing_mode: How indices are assigned to shards.
            pad_to_equal_size_across_shards: If ``True``, all shards report the same length. Padding repeats
                the last item of the dataset.

        Raises:
            ValueError: If ``dataset`` does not implement ``__len__``.
        """
        if not isinstance(dataset, Sized):
            raise ValueError(f"Dataset ({type(dataset).__name__}) must implement __len__() to be sharded.")

        self._dataset = dataset

        self._total_shards = total_shards
        self._current_shard = current_shard

        self._indexing_mode = indexing_mode
        self._pad_to_equal_size_across_shards = pad_to_equal_size_across_shards

    def _compute_real_index_sequential(self, index: int) -> int:
        return index * self._total_shards + self._current_shard

    def _get_base_index_unsafe(self, index: int) -> int:
        """Computes the underlying dataset index for a shard index, without bounds checking.

        Returns:
            The index in the underlying dataset.

        Raises:
            ValueError: If the indexing mode is unknown.
        """
        match self._indexing_mode:
            case ShardIndexingMode.sequential:
                base_index = index * self._total_shards + self._current_shard

                return base_index
            case ShardIndexingMode.chunked:
                ceil_len = math.ceil(len(self._dataset) / self._total_shards)
                shard_start_offset = ceil_len * self._current_shard

                return shard_start_offset + index
            case _:
                raise ValueError(f"Unknown shard indexing mode ({self._indexing_mode}).")

    def __getitem__(self, index: int) -> _T_co:
        """Returns the item at an index relative to this shard.

        If the index points past the end of the dataset, which happens only with padding, returns the last item of
        the dataset.

        Args:
            index: The index relative to this shard.

        Returns:
            The data item.
        """
        base_index = self._get_base_index_unsafe(index)
        if base_index >= len(self._dataset):
            base_index = len(self._dataset) - 1
        return self._dataset[base_index]

    def __len__(self) -> int:
        """Returns the number of items in this shard.

        If ``pad_to_equal_size_across_shards`` is ``True``, returns the maximum shard length.

        Returns:
            The number of items in the shard.

        Raises:
            ValueError: If the indexing mode is unknown.
        """
        ceil_len = math.ceil(len(self._dataset) / self._total_shards)

        if self._pad_to_equal_size_across_shards:
            return ceil_len

        shards_remainder = len(self._dataset) % self._total_shards
        match self._indexing_mode:
            case ShardIndexingMode.sequential:
                shards_full = len(self._dataset) // self._total_shards
                return shards_full + 1 if self._current_shard < shards_remainder else shards_full
            case ShardIndexingMode.chunked:
                is_shard_last = self._current_shard == self._total_shards - 1
                if not is_shard_last or shards_remainder == 0:
                    return ceil_len
                else:
                    return ceil_len - (self._total_shards - shards_remainder)
            case _:
                raise ValueError(f"Unknown shard indexing mode ({self._indexing_mode}).")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if isinstance(self._dataset, Stateful):
            self._dataset.load_state_dict(state_dict["dataset"])

        # Shard contents depend on total_shards, so a checkpoint from another shard count is not valid here.
        if state_dict["total_shards"] != self._total_shards:
            raise ValueError(
                f"Shard count mismatch: the checkpoint has total_shards ({state_dict['total_shards']}), "
                f"but this dataset has total_shards ({self._total_shards}). Resume with the same number of shards."
            )
        self._total_shards = state_dict["total_shards"]

        self._current_shard = state_dict["current_shard"]

    def state_dict(self) -> dict[str, Any]:
        dct: dict[str, Any] = {"total_shards": self._total_shards, "current_shard": self._current_shard}
        if isinstance(self._dataset, Stateful):
            dct["dataset"] = self._dataset.state_dict()
        return dct


def shard_dataset_data_parallel(
    dataset: Dataset[_T_co],
    dist_context: DistributedContext,
    indexing_mode: ShardIndexingMode = ShardIndexingMode.sequential,
    pad_to_equal_size_across_shards: bool = True,
) -> Dataset[_T_co]:
    """Wraps a dataset into a ``ShardedDataset`` over the data-parallel ranks.

    The shard count and index come from the ``"dp"`` dimension of the batch domain mesh. Without distributed
    training, the dataset has a single shard.

    Args:
        dataset: The source dataset to shard.
        dist_context: The distributed context.
        indexing_mode: How indices are assigned to shards.
        pad_to_equal_size_across_shards: If ``True``, all shards have the same length.

    Returns:
        A dataset instance representing the local shard.
    """
    if dist_context.mesh_params.is_distributed:
        dp_mesh = dist_context.mesh_for(BATCH_DOMAIN)["dp"]
        n_shards = dp_mesh.size()
        current_shard = dp_mesh.get_local_rank()
    else:
        n_shards = 1
        current_shard = 0

    return ShardedDataset(
        dataset=dataset,
        total_shards=n_shards,
        current_shard=current_shard,
        indexing_mode=indexing_mode,
        pad_to_equal_size_across_shards=pad_to_equal_size_across_shards,
    )
