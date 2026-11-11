import pytest
import torch
from d9d.dist_context import BASE_DOMAIN, DENSE_DOMAIN, BaseDim, DenseDim, DenseDomainConfig, DeviceMeshParameters
from d9d.parallel import parallelize_fsdp, parallelize_hsdp, parallelize_replicate
from d9d.parallel.placement import check_data_shard_dim
from torch import nn
from torch.distributed.tensor import DTensor

from d9d_dev.parallel.runtime_step import step_with_runtime


class _Aliased(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(16, device="cuda"))
        self.alias = self.weight
        self.child = nn.Module()
        self.child.register_parameter("weight", self.weight)

    def forward(self, inputs, shared):
        return inputs * (self.weight + self.alias + self.child.weight)


@pytest.mark.distributed
@pytest.mark.parametrize("strategy", ["fsdp", "hsdp", "ddp"])
def test_placement_preserves_all_parameter_aliases(dist_ctx_factory, strategy):
    ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4)))
    mesh = ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE, DenseDim.DP_CP_SHARD]
    model = _Aliased()
    match strategy:
        case "fsdp":
            parallelize_fsdp(model, mesh[DenseDim.DP_CP_SHARD])
        case "hsdp":
            parallelize_hsdp(model, mesh, shard_dim=DenseDim.DP_CP_SHARD)
        case "ddp":
            parallelize_replicate(model, mesh)
        case _:
            raise ValueError(f"Unknown strategy ({strategy}).")

    storage = model.weight
    assert isinstance(storage, DTensor)
    assert model.alias is storage
    assert model.child.weight is storage
    assert len(list(model.parameters())) == 1

    def _step():
        assert model.alias is model.weight
        assert model.child.weight is model.weight
        model(torch.ones(16, device="cuda"), None).sum().backward()

    runtime = step_with_runtime(model, _step)

    replicas = 4 if strategy == "fsdp" else 8
    torch.testing.assert_close(storage.grad.to_local(), torch.full_like(storage.to_local(), 3 * replicas))
    assert model.weight is storage
    assert model.alias is storage
    assert model.child.weight is storage
    runtime.unbind()


@pytest.mark.distributed
def test_hsdp_rejects_other_shard_dims(dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))
    mesh = ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE, DenseDim.DP_CP_SHARD]
    model = nn.Linear(16, 8, device="cuda")
    storage = model.weight

    with pytest.raises(ValueError, match="not a data shard dimension"):
        parallelize_hsdp(model, mesh, shard_dim=DenseDim.DP_REPLICATE)

    assert model.weight is storage


@pytest.mark.distributed
def test_hsdp_rejects_a_mesh_without_the_shard_dim(dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))
    mesh = ctx.mesh_for(BASE_DOMAIN)[BaseDim.DP, BaseDim.CP]
    model = nn.Linear(16, 8, device="cuda")
    storage = model.weight

    with pytest.raises(ValueError, match="has no dimension"):
        parallelize_hsdp(model, mesh, shard_dim=DenseDim.DP_CP_SHARD)

    assert model.weight is storage


@pytest.mark.local
def test_check_data_shard_dim():
    check_data_shard_dim("dp_cp_shard")
    with pytest.raises(ValueError, match="not a data shard dimension"):
        check_data_shard_dim("tp")
