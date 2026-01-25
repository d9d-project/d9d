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

    assert spec == {"a": None, "b": None, "c": (None, None)}


@pytest.mark.local
def test_shard_spec_on_dim():
    tree = {
        "valid": torch.randn(10, 10),
        "too_small_rank": torch.randn(5),
        "scalar": torch.tensor(1.0),
        "not_tensor": "string_val",
        "valid_list": [1, 2, 3, 4]
    }

    spec = shard_spec_on_dim(tree, 0)
    assert spec == {
        "valid": Shard(0),
        "too_small_rank": Shard(0),
        "scalar": None,
        "not_tensor": None,
        "valid_list": Shard(0)
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
def test_simple_list_sharding_even():
    data = [1, 2, 3, 4, 5, 6]
    spec = Shard(0)
    num_shards = 3

    shards = shard_tree(data, spec, num_shards, enforce_even_split=True)
    assert len(shards) == 3
    assert shards[0] == [1, 2]
    assert shards[1] == [3, 4]
    assert shards[2] == [5, 6]

    restored = unshard_tree(shards, spec)
    assert restored == data


@pytest.mark.local
def test_simple_list_sharding_uneven():
    data = ["a", "b", "c", "d", "e"]
    spec = Shard(0)
    num_shards = 2

    # 5 items / 2 shards = 3 and 2
    shards = shard_tree(data, spec, num_shards, enforce_even_split=False)

    assert len(shards) == 2
    assert len(shards[0]) + len(shards[1]) == 5

    # Check specific split logic
    assert shards[0] == ["a", "b", "c"]
    assert shards[1] == ["d", "e"]

    restored = unshard_tree(shards, spec)
    assert restored == data


@pytest.mark.local
def test_list_of_objects_roundtrip():
    o1, o2, o3 = {"id": 1}, {"id": 2}, {"id": 3}
    data = [o1, o2, o3]
    spec = Shard(0)

    shards = shard_tree(data, spec, 3, enforce_even_split=True)
    assert shards[0] == [o1]

    restored = unshard_tree(shards, spec)
    assert restored[0] is o1  # Check identity preservation


@pytest.mark.local
def test_enforce_even_split_raises():
    spec = Shard(0)

    with pytest.raises(ValueError, match="perfectly divisible"):
        shard_tree(torch.randn(10), spec, 3, enforce_even_split=True)

    with pytest.raises(ValueError, match="perfectly divisible"):
        shard_tree([1, 2, 3, 4], spec, 3, enforce_even_split=True)


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
        "model": {
            "weights": torch.randn(4, 4),
            "vocab_list": ["a", "b", "c", "d"],
            "names": ["max", "mike"]
        },
        "meta": 123,
    }

    spec = {
        "model": {
            "weights": Shard(1),
            "vocab_list": Shard(0),
            "names": None
        },
        "meta": None
    }

    num_shards = 2
    sharded_results = shard_tree(tree, spec, num_shards, enforce_even_split=True)
    assert len(sharded_results) == 2

    # Check Rank 0
    r0 = sharded_results[0]
    assert r0["model"]["weights"].shape == (4, 2)
    assert r0["model"]["vocab_list"] == ["a", "b"]
    assert r0["model"]["names"] == ["max", "mike"]
    assert r0["meta"] == 123

    # Check Rank 1
    r1 = sharded_results[1]
    assert r1["model"]["weights"].shape == (4, 2)
    assert r1["model"]["vocab_list"] == ["c", "d"]
    assert r1["model"]["names"] == ["max", "mike"]
    assert r1["meta"] == 123

    # Roundtrip
    restored = unshard_tree(sharded_results, spec)
    assert torch.equal(restored["model"]["weights"], tree["model"]["weights"])
    assert restored["model"]["vocab_list"] == ["a", "b", "c", "d"]
    assert restored["model"]["names"] == ["max", "mike"]
    assert restored["meta"] == 123


@pytest.mark.local
def test_shard_structure_mismatch():
    tree = [torch.randn(2), torch.randn(2)]
    spec = [Shard(0)]

    with pytest.raises(ValueError, match="structure does not match"):
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

    with pytest.raises(ValueError, match="Structure mismatch"):
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


@pytest.mark.local
def test_list_invalid_dim_sharding():
    data = [1, 2, 3, 4]
    spec = Shard(1)
    with pytest.raises(ValueError, match="dim 0"):
        shard_tree(data, spec, 2, enforce_even_split=False)
