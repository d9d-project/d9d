import dataclasses
from typing import Any

import pytest
import torch
from d9d.core.sharding import ShardingSpecLeaf, SpecShard
from d9d.internals.pipeline_state import PipelineStateHandler
from torch.testing import assert_close


@dataclasses.dataclass
class TestCase:
    num_shards: int
    global_state: dict[str, Any]
    sharded_state: dict[int, dict[str, Any]]
    sharding_spec: dict[str, ShardingSpecLeaf]


_TEST_CASES = [
    # 1D simple
    TestCase(
        num_shards=2,
        global_state={
            "loss": torch.tensor([0.5, 1.5])
        },
        sharded_state={
            0: {"loss": torch.tensor([0.5])},
            1: {"loss": torch.tensor([1.5])}
        },
        sharding_spec={"loss": SpecShard(0)}
    ),
    TestCase(
        num_shards=2,
        global_state={
            "loss": torch.tensor([0.5, 1.5])
        },
        sharded_state={
            0: {"loss": torch.tensor([0.5])},
            1: {"loss": torch.tensor([1.5])}
        },
        sharding_spec={}
    ),
    # 1D multiple elements
    TestCase(
        num_shards=2,
        global_state={
            "data": torch.tensor([0., 1., 2., 3.])
        },
        sharded_state={
            0: {"data": torch.tensor([0., 1.])},
            1: {"data": torch.tensor([2., 3.])}
        },
        sharding_spec={"data": SpecShard(0)}
    ),
    TestCase(
        num_shards=2,
        global_state={
            "data": torch.tensor([0., 1., 2., 3.])
        },
        sharded_state={
            0: {"data": torch.tensor([0., 1.])},
            1: {"data": torch.tensor([2., 3.])}
        },
        sharding_spec={}
    ),
    # 2D default
    TestCase(
        num_shards=2,
        global_state={
            "matrix": torch.tensor([[1., 2.], [3., 4.]])
        },
        sharded_state={
            0: {"matrix": torch.tensor([[1., 2.]])},
            1: {"matrix": torch.tensor([[3., 4.]])}
        },
        sharding_spec={"matrix": SpecShard(0)}
    ),
    # 2D col
    TestCase(
        num_shards=2,
        global_state={
            "matrix": torch.tensor([[1., 2.], [3., 4.]])
        },
        sharded_state={
            0: {"matrix": torch.tensor([[1.], [3.]])},
            1: {"matrix": torch.tensor([[2.], [4.]])}
        },
        sharding_spec={"matrix": SpecShard(1)}
    ),
    # List Split
    TestCase(
        num_shards=2,
        global_state={
            "ids": [10, 20, 30, 40]
        },
        sharded_state={
            0: {"ids": [10, 20]},
            1: {"ids": [30, 40]}
        },
        sharding_spec={}
    ),
    TestCase(
        num_shards=2,
        global_state={
            "ids": [10, 20, 30, 40]
        },
        sharded_state={
            0: {"ids": [10, 20]},
            1: {"ids": [30, 40]}
        },
        sharding_spec={"ids": SpecShard(0)}
    ),
    TestCase(
        num_shards=2,
        global_state={
            "metrics": torch.tensor([10., 20.])
        },
        sharded_state={
            0: {"metrics": torch.tensor(10.)},
            1: {"metrics": torch.tensor(20.)}
        },
        sharding_spec={"metrics": SpecShard(dim=0, do_stack=True)}
    ),
    TestCase(
        num_shards=2,
        global_state={
            "batch_embeddings": torch.tensor([
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2]
            ])  # Shape (2, 3)
        },
        sharded_state={
            0: {"batch_embeddings": torch.tensor([1.0, 1.1, 1.2])},
            1: {"batch_embeddings": torch.tensor([2.0, 2.1, 2.2])}
        },
        sharding_spec={"batch_embeddings": SpecShard(dim=0, do_stack=True)}
    ),
    TestCase(
        num_shards=2,
        global_state={
            "items": [15, 12]
        },
        sharded_state={
            0: {"items": 15},
            1: {"items": 12}
        },
        sharding_spec={"items": SpecShard(dim=0, do_stack=True)}
    ),
]

_TEST_CASES_WRITE_SHARD_READ_GLOBAL = [
    TestCase(
        num_shards=2,
        global_state={
            "batch_embeddings": torch.tensor([0.5, 1.0])
        },
        sharded_state={
            0: {"batch_embeddings": torch.tensor(0.5)},
            1: {"batch_embeddings": torch.tensor(1.0)}
        },
        sharding_spec={}
    ),
    TestCase(
        num_shards=2,
        global_state={
            "items": [15, 12]
        },
        sharded_state={
            0: {"items": 15},
            1: {"items": 12}
        },
        sharding_spec={}
    ),
]

_TEST_CASES_WRITE_GLOBAL_READ_SHARD = [
    TestCase(
        num_shards=2,
        global_state={
            "batch_embeddings": torch.tensor([0.5, 1.0])
        },
        sharded_state={
            0: {"batch_embeddings": torch.tensor([0.5])},
            1: {"batch_embeddings": torch.tensor([1.0])}
        },
        sharding_spec={}
    ),
    TestCase(
        num_shards=2,
        global_state={
            "items": [15, 12]
        },
        sharded_state={
            0: {"items": [15]},
            1: {"items": [12]}
        },
        sharding_spec={}
    ),
]


@pytest.mark.local
@pytest.mark.parametrize("case", _TEST_CASES)
def test_write_global_read_global(case: TestCase):
    handler = PipelineStateHandler(sharding_spec=case.sharding_spec, num_shards=case.num_shards)

    # Write
    for k, v in case.global_state.items():
        handler.global_state()[k] = v

    # Read & Verify
    for k, expected_v in case.global_state.items():
        actual_v = handler.global_state()[k]
        assert_close(actual_v, expected_v)


@pytest.mark.local
@pytest.mark.parametrize("case", _TEST_CASES)
def test_write_sharded_read_sharded(case: TestCase):
    handler = PipelineStateHandler(sharding_spec=case.sharding_spec, num_shards=case.num_shards)

    # Write all shards
    for shard_id, data_dict in case.sharded_state.items():
        for k, v in data_dict.items():
            handler.sharded_state(shard_id)[k] = v

    # Read & Verify all shards in definition
    for shard_id, expected_data in case.sharded_state.items():
        shard_view = handler.sharded_state(shard_id)

        for k, expected_v in expected_data.items():
            assert_close(shard_view[k], expected_v)


@pytest.mark.local
@pytest.mark.parametrize("case", _TEST_CASES + _TEST_CASES_WRITE_GLOBAL_READ_SHARD)
def test_write_global_read_sharded(case: TestCase):
    handler = PipelineStateHandler(sharding_spec=case.sharding_spec, num_shards=case.num_shards)

    # Write Global
    for k, v in case.global_state.items():
        handler.global_state()[k] = v

    # Read Shards
    for shard_id, expected_data in case.sharded_state.items():
        shard_view = handler.sharded_state(shard_id)

        for k, expected_v in expected_data.items():
            assert_close(shard_view[k], expected_v)


@pytest.mark.local
@pytest.mark.parametrize("case", _TEST_CASES + _TEST_CASES_WRITE_SHARD_READ_GLOBAL)
def test_write_sharded_read_global(case: TestCase):
    handler = PipelineStateHandler(sharding_spec=case.sharding_spec, num_shards=case.num_shards)

    # Write Shards
    for shard_id, data_dict in case.sharded_state.items():
        for k, v in data_dict.items():
            handler.sharded_state(shard_id)[k] = v

    # Read Global
    global_view = handler.global_state()

    for k, expected_v in case.global_state.items():
        assert_close(global_view[k], expected_v)


@pytest.mark.local
def test_detach_mechanics():
    handler = PipelineStateHandler(sharding_spec={}, num_shards=2)

    # Global Write Detach
    t_grad = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    handler.global_state()["grad_tensor"] = t_grad

    stored = handler.global_state()["grad_tensor"]
    assert stored.requires_grad is False
    assert stored.grad_fn is None
    assert_close(stored, t_grad.detach())

    # Sharded Write Detach
    t_grad_shard = torch.tensor([5.], requires_grad=True)
    handler.sharded_state(0)["grad_shard"] = t_grad_shard

    stored_shard = handler.sharded_state(0)["grad_shard"]
    assert stored_shard.requires_grad is False
    assert stored_shard.grad_fn is None
    assert_close(stored_shard, t_grad_shard.detach())
