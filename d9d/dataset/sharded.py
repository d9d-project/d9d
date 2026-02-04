import math
from collections.abc import Sized
from enum import StrEnum
from typing import Any, TypeVar

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset

from d9d.core.dist_context import BATCH_DOMAIN, DistributedContext


class ShardIndexingMode(StrEnum):
    """
    Defines how the dataset is split across shards.

    Modes:
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
    """
    A dataset wrapper that acts as a view onto a specific shard of the underlying dataset.

    This is useful for Data Parallel training where each process should only see
    a subset of the data. It supports different indexing modes and optional padding
    to ensure all shards have equal length (preventing hangs in distributed collectives).
    """

    def __init__(
            self,
            dataset: Dataset[_T_co],
            total_shards: int,
            current_shard: int,
            indexing_mode: ShardIndexingMode,
            pad_to_equal_size_across_shards: bool
    ):
        """
        Constructs a ShardedDataset object.

        Args:
            dataset: The underlying dataset to shard.
            total_shards: The total number of shards (e.g., number of DP ranks).
            current_shard: The index of the current shard (e.g., current DP rank).
            indexing_mode: How indices are assigned to shards (sequential/round-robin or chunked).
            pad_to_equal_size_across_shards: If True, the length of the dataset will be padded
                so that all shards report the same length. The last standard element is repeated.
        """

        if not isinstance(dataset, Sized):
            raise ValueError("Dataset should implement __len__ method")

        self._dataset = dataset

        self._total_shards = total_shards
        self._current_shard = current_shard

        self._indexing_mode = indexing_mode
        self._pad_to_equal_size_across_shards = pad_to_equal_size_across_shards

    def _compute_real_index_sequential(self, index: int) -> int:
        return index * self._total_shards + self._current_shard

    def _get_base_index_unsafe(self, index: int) -> int:
        """
        Calculates the underlying dataset index for a given shard index,
        without boundary checking.
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
                raise ValueError(f"Unknown shard indexing mode: {self._indexing_mode}")

    def __getitem__(self, index: int) -> _T_co:
        """
        Retrieves an item from the underlying dataset mapping logic shard index to physical index.

        If padding is enabled and the index exceeds the valid data for this shard,
        the last item in the dataset is returned.

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
        """
        Returns the number of items in this specific shard.

        If `pad_to_equal_size_across_shards` is True, this returns the ceiling
        length (max length across all shards).
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

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if isinstance(self._dataset, Stateful):
            self._dataset.load_state_dict(state_dict["dataset"])

        # check whether env mismatched
        if state_dict["total_shards"] != self._total_shards:
            raise ValueError("Shard count mismatch")
        self._total_shards = state_dict["total_shards"]

        self._current_shard = state_dict["current_shard"]

    def state_dict(self) -> dict[str, Any]:
        dct: dict[str, Any] = {
            "total_shards": self._total_shards,
            "current_shard": self._current_shard
        }
        if isinstance(self._dataset, Stateful):
            dct["dataset"] = self._dataset.state_dict()
        return dct


def shard_dataset_data_parallel(
        dataset: Dataset[_T_co],
        dist_context: DistributedContext,
        indexing_mode: ShardIndexingMode = ShardIndexingMode.sequential,
        pad_to_equal_size_across_shards: bool = True
) -> Dataset[_T_co]:
    """
    Wraps a dataset into a ShardedDataset based on the Data Parallel dimension of the distributed context.

    This is a helper function to automatically determine the correct rank and world size
    from the 'dp' (Data Parallel) mesh dimension within the batch domain DeviceMesh.

    Args:
        dataset: The source dataset to shard.
        dist_context: The distributed context.
        indexing_mode: The strategy for splitting data indices (sequential/round-robin or chunked).
        pad_to_equal_size_across_shards: If True, ensures all shards have the same length by padding.

    Returns:
        A dataset instance representing the local shard.
    """

    dp_mesh = dist_context.mesh_for(BATCH_DOMAIN)["dp"]
    return ShardedDataset(
        dataset=dataset,
        total_shards=dp_mesh.size(),
        current_shard=dp_mesh.get_local_rank(),
        indexing_mode=indexing_mode,
        pad_to_equal_size_across_shards=pad_to_equal_size_across_shards
    )
