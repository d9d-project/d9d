import pytest
import torch
from d9d.core.dist_context import REGULAR_DOMAIN
from d9d.model_state.mapper import ModelStateMapper, StateGroup
from d9d.model_state.mapper.compose import (
    ModelStateMapperParallel,
    ModelStateMapperSequential,
    ModelStateMapperShard,
)
from d9d.model_state.mapper.leaf import (
    ModelStateMapperDistribute,
    ModelStateMapperGatherFullTensor,
    ModelStateMapperIdentity,
    ModelStateMapperRename,
    ModelStateMapperSelectChildModules,
    ModelStateMapperStackTensors,
)
from torch.distributed.tensor import DTensor, Shard


class MergeAddMapper(ModelStateMapper):
    """
    Test helper: Takes two tensors (a, b) and outputs one (sum = a + b).
    Simulates complex logic modifying tensors.
    """

    def __init__(self, in_a: str, in_b: str, out_sum: str):
        self.in_a = in_a
        self.in_b = in_b
        self.out_sum = out_sum

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self.in_a, self.in_b]), outputs=frozenset([self.out_sum]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {self.out_sum: group[self.in_a] + group[self.in_b]}


@pytest.mark.local
def test_leaf_identity():
    mapper = ModelStateMapperIdentity("foo")

    expected_groups = frozenset([
        StateGroup(inputs=frozenset(["foo"]), outputs=frozenset(["foo"]))
    ])
    assert mapper.state_dependency_groups() == expected_groups

    data = {"foo": torch.tensor(1.0)}
    res = mapper.apply(data)
    assert res.keys() == data.keys()
    assert res["foo"] is data["foo"]


@pytest.mark.local
def test_leaf_rename():
    mapper = ModelStateMapperRename("old", "new")

    expected_groups = frozenset([
        StateGroup(inputs=frozenset(["old"]), outputs=frozenset(["new"]))
    ])
    assert mapper.state_dependency_groups() == expected_groups

    res = mapper.apply({"old": torch.tensor(123)})
    assert res.keys() == {"new"}
    assert res["new"].item() == 123


@pytest.mark.local
def test_leaf_select_child():
    mapper = ModelStateMapperSelectChildModules(["w", "b"], "layer1")

    expected_groups = frozenset([
        StateGroup(inputs=frozenset(["layer1.w"]), outputs=frozenset(["w"])),
        StateGroup(inputs=frozenset(["layer1.b"]), outputs=frozenset(["b"]))
    ])
    assert mapper.state_dependency_groups() == expected_groups

    res = mapper.apply({"layer1.w": torch.tensor(5)})
    assert res.keys() == {"w"}
    assert res["w"].item() == 5


@pytest.mark.local
def test_parallel_composition():
    m1 = ModelStateMapperRename("a", "x")
    m2 = ModelStateMapperRename("b", "y")

    # Valid composition
    par = ModelStateMapperParallel([m1, m2])

    expected_groups = frozenset([
        StateGroup(inputs=frozenset(["a"]), outputs=frozenset(["x"])),
        StateGroup(inputs=frozenset(["b"]), outputs=frozenset(["y"]))
    ])
    assert par.state_dependency_groups() == expected_groups

    with pytest.raises(ValueError, match="undefined group"):
        par.apply({"a": torch.tensor(1), "b": torch.tensor(1)})

    res = par.apply({"a": torch.tensor(1)})
    assert res.keys() == {"x"}
    assert res["x"].item() == 1

    res = par.apply({"b": torch.tensor(1)})
    assert res.keys() == {"y"}
    assert res["y"].item() == 1

    # Collision detection
    m3 = ModelStateMapperRename("a", "z")  # 'a' is already consumed by m1
    with pytest.raises(ValueError, match="colliding input"):
        ModelStateMapperParallel([m1, m3])


@pytest.mark.local
def test_sequential_composition_gap_filling():
    # Chain: (a -> b) THEN (b -> c)
    # But 'x' is separate.
    # We want: Input(a, x) -> [Rename a->b] -> Intermediate(b, ?) -> [Identity b, Rename x->y] -> Output(b, y)

    # Scenario:
    # 1. Rename 'a' -> 'temp'
    # 2. Add 'temp' + 'b' -> 'result'
    # If we pass {'a', 'b'} into Sequential, standard logic should fill the gap for 'b' in step 1.

    m1 = ModelStateMapperRename("a", "temp")
    m2 = MergeAddMapper("temp", "b", "result")

    seq = ModelStateMapperSequential([m1, m2])

    # Check merged group: Inputs of first stage, Outputs of last stage.
    expected_groups = frozenset([
        StateGroup(inputs=frozenset(["a", "b"]), outputs=frozenset(["result"]))
    ])
    assert seq.state_dependency_groups() == expected_groups

    # Execution: 'b' should pass through m1 via auto-added Identity
    out = seq.apply({"a": torch.tensor(10), "b": torch.tensor(5)})
    assert out["result"].item() == 15


@pytest.mark.local
def test_sequential_composition_chaining():
    # a -> b -> c
    m1 = ModelStateMapperRename("a", "b")
    m2 = ModelStateMapperRename("b", "c")
    seq = ModelStateMapperSequential([m1, m2])

    expected_groups = frozenset([
        StateGroup(inputs=frozenset(["a"]), outputs=frozenset(["c"]))
    ])
    assert seq.state_dependency_groups() == expected_groups

    assert seq.apply({"a": torch.tensor(1)})["c"].item() == 1


@pytest.mark.local
def test_shard_mapper():
    # Create 4 independent tasks
    mappers = [ModelStateMapperRename(f"in_{i}", f"out_{i}") for i in range(4)]
    par = ModelStateMapperParallel(mappers)

    # Shard 1 of 2 (Should get index 0 and 2)
    s0 = ModelStateMapperShard(par, total_shards=2, current_shard=0)

    expect_s0_groups = frozenset([
        StateGroup(inputs=frozenset(["in_0"]), outputs=frozenset(["out_0"])),
        StateGroup(inputs=frozenset(["in_2"]), outputs=frozenset(["out_2"]))
    ])
    assert s0.state_dependency_groups() == expect_s0_groups

    # Shard 2 of 2 (Should get index 1 and 3)
    s1 = ModelStateMapperShard(par, total_shards=2, current_shard=1)

    expect_s1_groups = frozenset([
        StateGroup(inputs=frozenset(["in_1"]), outputs=frozenset(["out_1"])),
        StateGroup(inputs=frozenset(["in_3"]), outputs=frozenset(["out_3"]))
    ])
    assert s1.state_dependency_groups() == expect_s1_groups


@pytest.mark.local
def test_stack_tensors_mapper():
    mapper = ModelStateMapperStackTensors(["a", "b", "c"], "stacked", stack_dim=0)

    expected_groups = frozenset([
        StateGroup(inputs=frozenset(["a", "b", "c"]), outputs=frozenset(["stacked"]))
    ])
    assert mapper.state_dependency_groups() == expected_groups

    t1 = torch.tensor([1, 1])
    t2 = torch.tensor([2, 2])
    t3 = torch.tensor([3, 3])

    res = mapper.apply({"a": t1, "b": t2, "c": t3})
    assert res.keys() == {"stacked"}
    assert res["stacked"].shape == (3, 2)
    assert torch.equal(res["stacked"][0], t1)
    assert torch.equal(res["stacked"][2], t3)


@pytest.mark.distributed
def test_dtensor_mapper_logic(dist_ctx_pp4_dpr2):
    mesh = dist_ctx_pp4_dpr2.mesh_for(REGULAR_DOMAIN)
    dp_mesh = mesh["dp_replicate"]

    # 1. Test Distribute Mapper
    full_tensor = torch.arange(8, device="cuda", dtype=torch.float32).reshape(4, 2)
    mapper_dist = ModelStateMapperDistribute("w", device_mesh=dp_mesh, placements=[Shard(0)])

    res = mapper_dist.apply({"w": full_tensor})
    assert isinstance(res["w"], DTensor)
    assert res["w"].to_local().shape == (2, 2)
    assert res["w"].shape == (4, 2)
    assert res["w"].placements == (Shard(0),)
    assert res["w"].device_mesh == dp_mesh

    # 2. Test Gather Mapper
    mapper_gather = ModelStateMapperGatherFullTensor("w")
    res_back = mapper_gather.apply({"w": res["w"]})

    assert isinstance(res_back["w"], torch.Tensor)
    assert not isinstance(res_back["w"], DTensor)

    # Verify content
    assert torch.equal(res_back["w"], full_tensor)
