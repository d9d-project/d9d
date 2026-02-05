from typing import Any, cast

import pytest
from d9d.core.dist_context import DeviceMeshParameters
from d9d.dataset import ShardedDataset, ShardIndexingMode, shard_dataset_data_parallel
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, length: int):
        self.data = list(range(length))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        return self.data[index]


class StatefulSimpleDataset(SimpleDataset, Stateful):
    def __init__(self, length: int):
        super().__init__(length)
        self.internal_state = "initial"

    def state_dict(self) -> dict[str, Any]:
        return {"internal_state": self.internal_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.internal_state = state_dict["internal_state"]


@pytest.mark.local
@pytest.mark.parametrize(
    ("num_items", "num_shards", "indexing_mode", "pad", "expect_data"),
    [
        (4, 2, ShardIndexingMode.sequential, False, [[0, 2], [1, 3]]),
        (5, 2, ShardIndexingMode.sequential, False, [[0, 2, 4], [1, 3]]),
        (5, 2, ShardIndexingMode.sequential, True, [[0, 2, 4], [1, 3, 4]]),
        (5, 2, ShardIndexingMode.chunked, False, [[0, 1, 2], [3, 4]]),
        (5, 2, ShardIndexingMode.chunked, True, [[0, 1, 2], [3, 4, 4]]),
        (1, 3, ShardIndexingMode.sequential, False, [[0], [], []]),
        (1, 3, ShardIndexingMode.sequential, True, [[0], [0], [0]]),
    ],
)
def test_sharding_ok(num_items, num_shards, expect_data, indexing_mode, pad):
    dataset = SimpleDataset(num_items)

    # Shard 0
    for shard_idx in range(num_shards):
        shard_dataset = ShardedDataset(
            dataset,
            total_shards=num_shards,
            current_shard=shard_idx,
            indexing_mode=indexing_mode,
            pad_to_equal_size_across_shards=pad,
        )
        expect_shard = expect_data[shard_idx]
        assert len(shard_dataset) == len(expect_shard)
        assert [shard_dataset[i] for i in range(len(expect_shard))] == expect_shard


@pytest.mark.local
def test_state_dict_save_load():
    dataset = StatefulSimpleDataset(10)
    dataset.internal_state = "modified"

    ds_wrapper = ShardedDataset(
        dataset,
        total_shards=4,
        current_shard=1,
        indexing_mode=ShardIndexingMode.sequential,
        pad_to_equal_size_across_shards=True,
    )

    state = ds_wrapper.state_dict()
    assert state["total_shards"] == 4
    assert state["current_shard"] == 1
    assert state["dataset"]["internal_state"] == "modified"

    # Restore to a new instance
    new_dataset = StatefulSimpleDataset(10)
    new_wrapper = ShardedDataset(
        new_dataset,
        total_shards=4,
        current_shard=0,  # Intentionally different initial params
        indexing_mode=ShardIndexingMode.sequential,
        pad_to_equal_size_across_shards=True,
    )

    new_wrapper.load_state_dict(state)

    assert new_wrapper._total_shards == 4
    assert new_wrapper._current_shard == 1
    assert new_dataset.internal_state == "modified"


@pytest.mark.local
def test_config_validation():
    """Verify standard validations."""
    # Test strict mismatch on total_shards
    dataset = SimpleDataset(10)
    ds = ShardedDataset(
        dataset,
        total_shards=4,
        current_shard=1,
        indexing_mode=ShardIndexingMode.sequential,
        pad_to_equal_size_across_shards=False,
    )
    state = ds.state_dict()
    state["total_shards"] = 5  # Tamper

    with pytest.raises(ValueError, match="Shard count mismatch"):
        ds.load_state_dict(state)

    # Test initialization with non-sized dataset
    class UnsizedDataset(Dataset):
        def __getitem__(self, index):
            return 0

    with pytest.raises(ValueError, match="Dataset should implement __len__"):
        ShardedDataset(UnsizedDataset(), 2, 0, ShardIndexingMode.sequential,
                       pad_to_equal_size_across_shards=False)


@pytest.mark.local
def test_shard_dataset_factory_non_distributed(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    dataset = SimpleDataset(100)
    sharded = shard_dataset_data_parallel(
        dataset,
        dist_ctx,
        indexing_mode=ShardIndexingMode.chunked,
        pad_to_equal_size_across_shards=True
    )

    assert isinstance(sharded, ShardedDataset)
    assert sharded._total_shards == 1
    assert sharded._current_shard == 0
    assert len(sharded) == 100
    assert sharded[99] == 99


@pytest.mark.distributed
def test_shard_dataset_factory_distributed(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    dataset = SimpleDataset(16)  # 16 items, 8 shards -> 2 items/shard

    sharded = shard_dataset_data_parallel(
        dataset,
        dist_ctx,
        indexing_mode=ShardIndexingMode.sequential,
        pad_to_equal_size_across_shards=False
    )

    sharded = cast(ShardedDataset, sharded)

    assert sharded._total_shards == 8

    dp_mesh = dist_ctx.mesh_for("batch")["dp"]
    expected_rank = dp_mesh.get_local_rank()

    assert sharded._current_shard == expected_rank

    local_items = [sharded[i] for i in range(len(sharded))]

    assert len(local_items) == 2

    assert local_items[0] == expected_rank
    assert local_items[1] == expected_rank + 8
