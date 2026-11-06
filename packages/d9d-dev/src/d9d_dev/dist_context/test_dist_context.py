import contextlib
from enum import StrEnum
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import (
    BASE_DOMAIN,
    DENSE_DOMAIN,
    EXPERT_DOMAIN,
    METRIC_DOMAIN,
    BaseDim,
    DenseDim,
    DenseDomainConfig,
    DeviceMeshDomain,
    DeviceMeshParameters,
    ExpertDim,
    ExpertDomainConfig,
    MeshDim,
    MetricDim,
    init_dist_context,
)
from pydantic import ValidationError
from torch.distributed.constants import default_pg_nccl_timeout


class _DataContextParallelDim(StrEnum):
    PP = "pp"
    DP_CP = "dp_cp"
    TP = "tp"


class _DataContextParallelDomain(DeviceMeshDomain[_DataContextParallelDim]):
    @property
    def name(self) -> str:
        return "dp_cp"

    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[_DataContextParallelDim], ...]:
        return (
            MeshDim(_DataContextParallelDim.PP, params.pipeline_parallel),
            MeshDim(_DataContextParallelDim.DP_CP, params.data_parallel * params.context_parallel),
            MeshDim(_DataContextParallelDim.TP, params.tensor_parallel),
        )


class _FixedDomain(DeviceMeshDomain[_DataContextParallelDim]):
    def __init__(self, name: str, dims: tuple[MeshDim[_DataContextParallelDim], ...]):
        self._name = name
        self._dims = dims

    @property
    def name(self) -> str:
        return self._name

    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[_DataContextParallelDim], ...]:
        return self._dims


DP_CP_DOMAIN = _DataContextParallelDomain()


@pytest.mark.local
@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"data_parallel": 4, "dense": {"data_parallel_shard": 3}}, "data-parallel shard size"),
        ({"context_parallel": 4, "dense": {"context_parallel_shard": 8}}, "context-parallel shard size"),
        # The expert-parallel degree divides dp * cp * tp but not dp * cp.
        ({"data_parallel": 2, "tensor_parallel": 2, "expert": {"expert_parallel": 4}}, "expert-parallel degree"),
        ({"data_parallel_replicate": 2}, "Extra inputs are not permitted"),
        ({"expert_parallel": 2}, "Extra inputs are not permitted"),
        ({"pipeline_parallel": 0}, "greater than 0"),
    ],
)
def test_params_validation(config, match):
    with pytest.raises(ValidationError, match=match):
        DeviceMeshParameters.model_validate(config)


@pytest.mark.local
def test_local_context():
    with init_dist_context(DeviceMeshParameters(), extra_domains=(DP_CP_DOMAIN,)) as ctx:
        assert not ctx.is_distributed
        assert (ctx.rank, ctx.world_size, ctx.local_rank) == (0, 1, 0)
        assert ctx.is_main_process
        assert ctx.is_local_main_process
        assert ctx.current_device == torch.device("cuda", 0)
        assert not dist.is_initialized()

        for domain, dims in (
            (BASE_DOMAIN, BaseDim),
            (METRIC_DOMAIN, MetricDim),
            (DP_CP_DOMAIN, _DataContextParallelDim),
        ):
            for dim in dims:
                assert ctx.coordinate(domain, dim) == 0
                assert ctx.dim_size(domain, dim) == 1

        with pytest.raises(ValueError, match="no parallelism"):
            ctx.mesh_for(BASE_DOMAIN)
        with pytest.raises(ValueError, match=r"mesh\.expert"):
            ctx.coordinate(EXPERT_DOMAIN, ExpertDim.EP_SHARD)
        with pytest.raises(ValueError, match="extra_domains"):
            ctx.coordinate(_DataContextParallelDomain(), _DataContextParallelDim.DP_CP)
        with pytest.raises(ValueError, match="no dimension"):
            ctx.coordinate(BASE_DOMAIN, DenseDim.DP_REPLICATE)

        ctx.set_timeout(10.0)
        ctx.wait_world()


@pytest.fixture
def restore_current_gpu():
    device = torch.cuda.current_device()
    yield
    torch.cuda.set_device(device)


@pytest.mark.local
@pytest.mark.skipif(torch.cuda.device_count() != 2, reason="The cases assume 2 visible GPUs.")
@pytest.mark.parametrize(
    ("local_world_size", "local_rank", "expected"),
    [
        (None, 1, 1),
        (2, 1, 1),
        (8, 3, 0),
        (8, 4, 1),
        (6, 5, 1),
        (3, 0, "split evenly"),
        (None, 2, "its own GPU"),
    ],
)
def test_gpu_assignment(monkeypatch, restore_current_gpu, local_world_size, local_rank, expected):
    # Without WORLD_SIZE, the context has one rank and assigns the GPU before it needs a process group.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    if local_world_size is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world_size))

    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected), init_dist_context(DeviceMeshParameters()):
            pass
        return
    with init_dist_context(DeviceMeshParameters()) as ctx:
        assert ctx.local_rank == local_rank
        assert ctx.current_device == torch.device("cuda", expected)
        assert torch.cuda.current_device() == expected


@pytest.mark.local
@pytest.mark.parametrize(
    ("params", "extra_domains", "match"),
    [
        (DeviceMeshParameters(data_parallel=2), (), "world size"),
        (DeviceMeshParameters(), (_FixedDomain("base", ()),), "domain names"),
        (
            DeviceMeshParameters(),
            (_FixedDomain("twice", (MeshDim(_DataContextParallelDim.PP, 1), MeshDim(_DataContextParallelDim.PP, 1))),),
            "dimension names",
        ),
        (DeviceMeshParameters(), (_FixedDomain("big", (MeshDim(_DataContextParallelDim.PP, 2),)),), "multiply"),
        (DeviceMeshParameters(), (_FixedDomain("zero", (MeshDim(_DataContextParallelDim.PP, 0),)),), "positive"),
    ],
)
def test_init_validation(params, extra_domains, match):
    with pytest.raises(ValueError, match=match), init_dist_context(params, extra_domains):
        pass


@pytest.mark.distributed
@pytest.mark.parametrize(
    "params",
    [
        DeviceMeshParameters(data_parallel=8),
        DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4)),
        DeviceMeshParameters(
            pipeline_parallel=2,
            data_parallel=4,
            dense=DenseDomainConfig(data_parallel_shard=2),
            expert=ExpertDomainConfig(expert_parallel=2),
        ),
        DeviceMeshParameters(
            data_parallel=2,
            context_parallel=2,
            tensor_parallel=2,
            dense=DenseDomainConfig(data_parallel_shard=2, context_parallel_shard=2),
            expert=ExpertDomainConfig(expert_parallel=4),
        ),
        # Before the expert mesh had a tensor-parallel dimension, these degrees failed to build it.
        DeviceMeshParameters(
            data_parallel=4,
            tensor_parallel=2,
            dense=DenseDomainConfig(data_parallel_shard=2),
            expert=ExpertDomainConfig(expert_parallel=4),
        ),
    ],
)
def test_meshes(dist_ctx_factory, params):
    ctx = dist_ctx_factory(params, (DP_CP_DOMAIN,))
    domains = {
        BASE_DOMAIN: BaseDim,
        DENSE_DOMAIN: DenseDim,
        METRIC_DOMAIN: MetricDim,
        DP_CP_DOMAIN: _DataContextParallelDim,
    }
    if params.expert is not None:
        domains[EXPERT_DOMAIN] = ExpertDim

    for domain, dims in domains.items():
        mesh = ctx.mesh_for(domain)
        assert mesh.mesh_dim_names == tuple(dims)
        assert sorted(mesh.mesh.flatten().tolist()) == list(range(8))
        assert mesh.get_coordinate() == tuple(ctx.coordinate(domain, dim) for dim in dims)
        assert mesh.shape == tuple(ctx.dim_size(domain, dim) for dim in dims)
        # Each group holds the ranks of the row of the mesh through this rank, in mesh order.
        coordinate = mesh.get_coordinate()
        for index, dim in enumerate(dims):
            row = list(coordinate)
            row[index] = slice(None)
            assert dist.get_process_group_ranks(mesh.get_group(dim)) == mesh.mesh[tuple(row)].tolist()

    # Dimensions with the same ranks share their groups.
    base = ctx.mesh_for(BASE_DOMAIN)
    shared = [
        (DENSE_DOMAIN, DenseDim.PP, BaseDim.PP),
        (DENSE_DOMAIN, DenseDim.TP, BaseDim.TP),
        (METRIC_DOMAIN, MetricDim.TP, BaseDim.TP),
        (DP_CP_DOMAIN, _DataContextParallelDim.TP, BaseDim.TP),
    ]
    if params.expert is not None:
        shared.append((EXPERT_DOMAIN, ExpertDim.TP, BaseDim.TP))
    for domain, dim, base_dim in shared:
        assert ctx.mesh_for(domain).get_group(dim) is base.get_group(base_dim)
    assert ctx.mesh_for(METRIC_DOMAIN).get_group(MetricDim.PP_DP_CP).size() == 8 // params.tensor_parallel
    if params.data_parallel == 8:
        # A dimension of all ranks uses the default group, and all dimensions of size 1 share their groups.
        assert base.get_group(BaseDim.DP) is dist.group.WORLD
        assert base.get_group(BaseDim.PP) is base.get_group(BaseDim.CP)


@pytest.mark.distributed
def test_mesh_for_rejects_unknown_domain(dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8), (DP_CP_DOMAIN,))

    # The keys are the domain objects, so another object with the same name is not a domain of the context.
    with pytest.raises(ValueError, match="extra_domains"):
        ctx.mesh_for(_DataContextParallelDomain())
    with pytest.raises(ValueError, match=r"mesh\.expert"):
        ctx.mesh_for(EXPERT_DOMAIN)


@pytest.mark.distributed
def test_inconsistent_domains(dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))
    extra_domains = (DP_CP_DOMAIN,) if ctx.is_main_process else ()

    # Every rank raises, none hangs in the group creation.
    with (
        pytest.raises(ValueError, match="other domains than rank 0"),
        init_dist_context(DeviceMeshParameters(data_parallel=8), extra_domains),
    ):
        pass


@pytest.mark.distributed
def test_exit_keeps_foreign_default_group():
    with init_dist_context(DeviceMeshParameters(data_parallel=2, context_parallel=4)) as ctx:
        group = ctx.mesh_for(BASE_DOMAIN).get_group(BaseDim.CP)

    assert dist.is_initialized()
    # A destroyed group is no longer registered.
    with pytest.raises(KeyError):
        dist.get_process_group_ranks(group)


@pytest.mark.distributed
def test_error_in_block_keeps_groups():
    with (
        pytest.raises(RuntimeError, match="error in the block"),
        init_dist_context(DeviceMeshParameters(data_parallel=2, context_parallel=4)) as ctx,
    ):
        group = ctx.mesh_for(BASE_DOMAIN).get_group(BaseDim.CP)
        raise RuntimeError("error in the block")

    # Destroying the group would wait for ranks that can hang in a collective, so it stays usable.
    value = torch.ones(1, device="cuda")
    dist.all_reduce(value, group=group)
    torch.testing.assert_close(value, torch.full((1,), 4.0, device="cuda"))
    dist.destroy_process_group(group)


@pytest.mark.distributed
@pytest.mark.parametrize("raises", [False, True])
def test_exit_restores_dtensor_rng_tracker(raises):
    dtensor_random = torch.distributed.tensor._random
    previous = dtensor_random._rng_tracker

    with contextlib.suppress(RuntimeError), init_dist_context(DeviceMeshParameters(data_parallel=8)):
        assert isinstance(dtensor_random._rng_tracker, dtensor_random.OffsetBasedRNGTracker)
        assert dtensor_random._rng_tracker is not previous
        if raises:
            raise RuntimeError("error in the block")

    assert dtensor_random._rng_tracker is previous


@pytest.mark.distributed
def test_sync_primitives(dist_ctx_factory, shared_tmp_dir: Path):
    ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))

    assert ctx.is_distributed
    assert (ctx.rank, ctx.world_size) == (dist.get_rank(), 8)
    assert ctx.is_main_process == (ctx.rank == 0)
    assert ctx.is_local_main_process == (ctx.local_rank == 0)
    ranks_per_gpu = max(1, 8 // torch.cuda.device_count())
    assert ctx.current_device == torch.device("cuda", ctx.local_rank // ranks_per_gpu)
    assert torch.cuda.current_device() == ctx.current_device.index

    ctx.set_timeout(10.0)
    try:
        ctx.wait_world()
    finally:
        # set_timeout also changes the default process group, which every later test in the session shares
        ctx.set_timeout(default_pg_nccl_timeout.total_seconds())

    with ctx.main_process_first():
        check_file = shared_tmp_dir / "main_first"
        if ctx.is_main_process:
            assert not check_file.is_file()
            check_file.write_text("")
        else:
            assert check_file.is_file()

    with ctx.local_main_process_first():
        check_file = shared_tmp_dir / "local_main_first"
        if ctx.is_local_main_process:
            assert not check_file.is_file()
            check_file.write_text("")
        else:
            assert check_file.is_file()
