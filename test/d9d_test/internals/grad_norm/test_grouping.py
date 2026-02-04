import pytest
import torch
from d9d.core.dist_context import EXPERT_DOMAIN, DeviceMeshParameters
from d9d.internals.grad_norm import group_parameters_for_norm
from torch import nn
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


@pytest.mark.distributed
def test_grouping_mechanism(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(
        pipeline_parallel=4,
        expert_parallel=2,
        data_parallel_replicate=2
    ))
    ep_mesh = dist_ctx.mesh_for(EXPERT_DOMAIN)

    local_param = nn.Parameter(
        distribute_tensor(
            torch.randn(16, 16, device="cuda"),
            device_mesh=ep_mesh[["ep_replicate", "ep_shard"]],
            placements=[Replicate(), Replicate()]
        )
    )

    sharded_param = nn.Parameter(
        distribute_tensor(
            torch.randn(16, 16, device="cuda"),
            device_mesh=ep_mesh[["ep_replicate", "ep_shard"]],
            placements=[Replicate(), Shard(0)]
        )
    )

    parameters = [local_param, sharded_param]

    groups = group_parameters_for_norm(parameters)

    assert len(groups) == 2

    keys = list(groups.keys())

    first_group = keys[0]
    second_group = keys[1]

    assert first_group.shard_meshes is not None, "First group should be the sharded one"
    assert len(first_group.shard_meshes) == 1
    assert first_group.shard_meshes[0] == ep_mesh["ep_shard"]

    assert second_group.shard_meshes is None, "Second group should be the local one"

    assert groups[first_group][0] is sharded_param
    assert groups[second_group][0] is local_param
