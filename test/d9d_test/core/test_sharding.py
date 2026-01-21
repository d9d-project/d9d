import pytest
import torch
from d9d.core.sharding import (
    shard_spec_nothing,
    shard_spec_on_dim,
    shard_tree,
    unshard_tree,
)
from torch.distributed.tensor import Shard


@pytest.mark.local
def test_shard_spec_nothing():
    spec = shard_spec_nothing({"a": torch.randn(4), "b": [torch.randn(2, 2), 42], "c": (1, 2)})

    assert spec == {"a": None, "b": [None, None], "c": (None, None)}


@pytest.mark.local
def test_shard_spec_on_dim():
    tree = {
        "valid": torch.randn(10, 10),
        "too_small_rank": torch.randn(5),
        "scalar": torch.tensor(1.0),
        "not_tensor": "string_val",
    }

    spec = shard_spec_on_dim(tree, 0)
    assert spec == {
        "valid": Shard(0),
        "too_small_rank": Shard(0),
        "scalar": None,
        "not_tensor": None
    }

    with pytest.raises(ValueError):
        shard_spec_on_dim(tree, 1)


@pytest.mark.local
def test_simple_tensor_even_split():
    data = torch.randn(8, 4)
    spec = Shard(0)
    num_shards = 4

    shards = shard_tree(data, spec, num_shards, enforce_even_split=True)
    assert isinstance(shards, tuple)
    assert len(shards) == 4
    for s in shards:
        assert s.shape == (2, 4)

    restored = unshard_tree(shards, spec)
    assert torch.equal(data, restored)


@pytest.mark.local
def test_simple_tensor_uneven_split():
    data = torch.randn(10)
    spec = Shard(0)
    num_shards = 3

    shards = shard_tree(data, spec, num_shards, enforce_even_split=False)

    assert isinstance(shards, tuple)
    sizes = [s.numel() for s in shards]
    assert sum(sizes) == 10
    assert len(shards) == 3

    restored = unshard_tree(shards, spec)
    assert torch.equal(data, restored)


@pytest.mark.local
def test_enforce_even_split_raises():
    data = torch.randn(10)
    spec = Shard(0)
    with pytest.raises(ValueError, match="perfectly divisible"):
        shard_tree(data, spec, 3, enforce_even_split=True)


@pytest.mark.local
def test_replication():
    data = torch.randn(3, 3)
    spec = None
    num_shards = 2

    shards = shard_tree(data, spec, num_shards, enforce_even_split=True)
    assert len(shards) == 2
    assert torch.equal(shards[0], data)
    assert torch.equal(shards[1], data)

    restored = unshard_tree(shards, spec)
    assert torch.equal(restored, data)


@pytest.mark.local
def test_complex_tree_mixed_sharding():
    tree = {
        "model": {"layers": [torch.randn(4, 8), torch.randn(4, 8)], "bias": torch.randn(4)},
        "meta": {"id": 123},
    }

    spec = {"model": {"layers": [Shard(0), Shard(1)], "bias": None}, "meta": {"id": None}}

    num_shards = 2
    sharded_results = shard_tree(tree, spec, num_shards, enforce_even_split=True)

    assert len(sharded_results) == 2
    shards = sharded_results

    # Verify Rank 0 shapes
    for shard in shards:
        assert shard["model"]["layers"][0].shape == (2, 8)
        assert shard["model"]["layers"][1].shape == (4, 4)
        assert shard["model"]["bias"].shape == (4,)
        assert shard["meta"]["id"] == 123

    # Roundtrip
    restored = unshard_tree(sharded_results, spec)

    assert torch.equal(restored["model"]["layers"][0], tree["model"]["layers"][0])
    assert torch.equal(restored["model"]["layers"][1], tree["model"]["layers"][1])
    assert torch.equal(restored["model"]["bias"], tree["model"]["bias"])
    assert restored["meta"]["id"] == tree["meta"]["id"]


@pytest.mark.local
def test_shard_structure_mismatch():
    tree = [torch.randn(2), torch.randn(2)]
    spec = [Shard(0)]

    with pytest.raises(ValueError, match="Tree structure mismatch"):
        shard_tree(tree, spec, 2, enforce_even_split=False)


@pytest.mark.local
def test_unshard_empty_shards():
    """Cannot unshard empty list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        unshard_tree([], Shard(0))


@pytest.mark.local
def test_unshard_structure_mismatch_across_ranks():
    spec = [Shard(0), Shard(0)]

    shard0 = [torch.randn(1), torch.randn(1)]
    shard1 = [torch.randn(1)]  # Missing one element

    with pytest.raises(ValueError, match="Structure mismatch at rank 1"):
        unshard_tree([shard0, shard1], spec)


@pytest.mark.local
def test_invalid_shard_spec_type():
    data = torch.randn(4)
    spec = "invalid_string_spec"

    with pytest.raises(TypeError, match="Unknown sharding spec"):
        shard_tree(data, spec, 2, enforce_even_split=False)


@pytest.mark.local
def test_shard_non_tensor_with_shard_object():
    data = 123
    spec = Shard(0)

    with pytest.raises(TypeError, match="item was not a Tensor"):
        shard_tree(data, spec, 2, enforce_even_split=False)
