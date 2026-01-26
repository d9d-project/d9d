import dataclasses
from typing import Any

import pytest
import torch
from d9d.core.sharding import (
    ShardingSpec,
    SpecReplicate,
    SpecShard,
    shard_spec_nothing,
    shard_spec_on_dim,
    shard_tree,
    unshard_tree,
)
from torch.testing import assert_close


@pytest.mark.local
def test_shard_spec_nothing():
    tree = {"a": torch.randn(4), "b": [torch.randn(2, 2), 42], "c": (1, 2)}
    spec = shard_spec_nothing(tree)
    # shard_spec_nothing returns None (checking type implicitly via equality with structure of Nones)
    assert spec == {"a": SpecReplicate(), "b": SpecReplicate(), "c": (SpecReplicate(), SpecReplicate())}


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
        "valid": SpecShard(0),
        "too_small_rank": SpecShard(0),
        "scalar": SpecReplicate(),
        "not_tensor": SpecReplicate(),
        "valid_list": SpecShard(0)
    }

    with pytest.raises(ValueError, match="Cannot shard"):
        shard_spec_on_dim(tree, 1)


@dataclasses.dataclass
class RoundtripCase:
    input_tree: Any
    spec: ShardingSpec
    num_shards: int
    expected_shards: tuple[Any, ...]
    enforce_even_split: bool


_OBJ_1, _OBJ_2, _OBJ_3 = {"id": 1}, {"id": 2}, {"id": 3}

_ROUNDTRIP_CASES = [
    RoundtripCase(
        input_tree=torch.arange(8).reshape(8, 1).float(),
        spec=SpecShard(0),
        num_shards=4,
        enforce_even_split=True,
        expected_shards=(
            torch.tensor([[0.], [1.]]),
            torch.tensor([[2.], [3.]]),
            torch.tensor([[4.], [5.]]),
            torch.tensor([[6.], [7.]]),
        )
    ),
    RoundtripCase(
        input_tree=torch.arange(10).float(),
        spec=SpecShard(0),
        num_shards=3,
        enforce_even_split=False,
        expected_shards=(
            torch.tensor([0., 1., 2., 3.]),  # 4
            torch.tensor([4., 5., 6.]),  # 3
            torch.tensor([7., 8., 9.]),  # 3
        )
    ),
    RoundtripCase(
        input_tree=[1, 2, 3, 4, 5, 6],
        spec=SpecShard(0),
        num_shards=3,
        enforce_even_split=True,
        expected_shards=(
            [1, 2],
            [3, 4],
            [5, 6]
        )
    ),
    RoundtripCase(
        input_tree=[10, 20, 30, 40, 50],
        spec=SpecShard(0),
        num_shards=2,
        enforce_even_split=False,
        expected_shards=(
            [10, 20, 30],
            [40, 50]
        )
    ),
    RoundtripCase(
        input_tree=[_OBJ_1, _OBJ_2, _OBJ_3],
        spec=SpecShard(0),
        num_shards=3,
        enforce_even_split=True,
        expected_shards=(
            [_OBJ_1],
            [_OBJ_2],
            [_OBJ_3]
        )
    ),
    RoundtripCase(
        input_tree=torch.tensor([1., 2., 3.]),
        spec=SpecReplicate(),
        num_shards=2,
        expected_shards=(
            torch.tensor([1., 2., 3.]),
            torch.tensor([1., 2., 3.])
        ),
        enforce_even_split=False
    ),
    RoundtripCase(
        input_tree={
            "model": {
                "weights": torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]),  # 4x2
                "vocab": [5, 6, 7, 8],
                "cfg": {"dim": 1}
            },
            "opt": torch.tensor([0., 0.])
        },
        spec={
            "model": {
                "weights": SpecShard(0),
                "vocab": SpecShard(0),
                "cfg": SpecReplicate()
            },
            "opt": SpecReplicate()
        },
        num_shards=2,
        enforce_even_split=True,
        expected_shards=(
            # Rank 0
            {
                "model": {
                    "weights": torch.tensor([[1., 2.], [3., 4.]]),
                    "vocab": [5, 6],
                    "cfg": {"dim": 1}
                },
                "opt": torch.tensor([0., 0.])
            },
            # Rank 1
            {
                "model": {
                    "weights": torch.tensor([[5., 6.], [7., 8.]]),
                    "vocab": [7, 8],
                    "cfg": {"dim": 1}
                },
                "opt": torch.tensor([0., 0.])
            }
        )
    ),
    RoundtripCase(
        input_tree=torch.tensor([[1., 2.], [3., 4.], [5., 6.]]),  # 3x2
        spec=SpecShard(0, do_stack=True),
        num_shards=3,
        enforce_even_split=True,
        expected_shards=(
            torch.tensor([1., 2.]),  # Rank reduces: (2,)
            torch.tensor([3., 4.]),
            torch.tensor([5., 6.])
        )
    ),
    RoundtripCase(
        input_tree=[123, 456, 789],
        spec=SpecShard(0, do_stack=True),
        num_shards=3,
        enforce_even_split=True,
        expected_shards=(
            123, 456, 789
        )
    ),
    RoundtripCase(
        input_tree=torch.tensor([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.]
        ]),  # 2x4
        spec=SpecShard(1),  # Split Columns
        num_shards=2,
        enforce_even_split=True,
        expected_shards=(
            torch.tensor([[1., 2.], [5., 6.]]),
            torch.tensor([[3., 4.], [7., 8.]])
        )
    ),
    RoundtripCase(
        input_tree=torch.tensor([
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.]
        ]),  # 4x2
        spec=SpecShard(1, do_stack=True),
        num_shards=2,
        enforce_even_split=True,
        expected_shards=(
            torch.tensor([1., 3., 5., 7.]),
            torch.tensor([2., 4., 6., 8.])
        )
    ),
]


@pytest.mark.local
@pytest.mark.parametrize("case", _ROUNDTRIP_CASES)
def test_sharding_roundtrip(case: RoundtripCase):
    shards = shard_tree(
        case.input_tree,
        case.spec,
        case.num_shards,
        enforce_even_split=case.enforce_even_split
    )

    assert len(shards) == case.num_shards

    for actual, expected in zip(shards, case.expected_shards, strict=True):
        assert_close(actual, expected)

    restored = unshard_tree(shards, case.spec)

    assert_close(restored, case.input_tree)


@pytest.mark.local
def test_enforce_even_split_raises():
    spec = SpecShard(0)
    with pytest.raises(ValueError, match="perfectly divisible"):
        shard_tree(torch.randn(10), spec, 3, enforce_even_split=True)

    with pytest.raises(ValueError, match="perfectly divisible"):
        shard_tree([1, 2, 3, 4], spec, 3, enforce_even_split=True)


@pytest.mark.local
def test_shard_structure_mismatch():
    tree = [torch.randn(2), torch.randn(2)]
    spec = [SpecShard(0)]

    with pytest.raises(ValueError, match="structure does not match"):
        shard_tree(tree, spec, 2, enforce_even_split=False)


@pytest.mark.local
def test_unshard_empty_shards():
    """Cannot unshard empty list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        unshard_tree([], SpecShard(0))


@pytest.mark.local
def test_unshard_structure_mismatch_across_ranks():
    spec = [SpecShard(0), SpecShard(0)]
    shard0 = [torch.randn(1), torch.randn(1)]
    shard1 = [torch.randn(1)]

    with pytest.raises(ValueError):
        unshard_tree([shard0, shard1], spec)


@pytest.mark.local
def test_invalid_shard_spec_type():
    data = torch.randn(4)
    spec = "invalid_string_spec"

    with pytest.raises(TypeError, match="Unknown sharding spec"):
        shard_tree(data, spec, 2, enforce_even_split=False)


@pytest.mark.local
def test_shard_non_tensor_with_shard_object():
    data = 123  # Int cannot be sharded
    spec = SpecShard(0)

    with pytest.raises(TypeError, match="item was not a Tensor"):
        shard_tree(data, spec, 2, enforce_even_split=False)


@pytest.mark.local
def test_list_invalid_dim_sharding():
    data = [1, 2, 3, 4]
    spec = SpecShard(1)  # List only supports dim 0
    with pytest.raises(ValueError, match="dim 0"):
        shard_tree(data, spec, 2, enforce_even_split=False)
