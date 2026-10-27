from itertools import product
from math import prod

import pytest

from d9d_test.modules.model.meshes import DENSE_MODEL_MESHES, MOE_MODEL_MESHES

pytestmark = pytest.mark.local


@pytest.mark.parametrize(
    "mesh",
    [
        pytest.param(mesh, id=f"{model_kind}-{name}")
        for model_kind, meshes in (("dense", DENSE_MODEL_MESHES), ("moe", MOE_MODEL_MESHES))
        for name, mesh in meshes.items()
    ],
)
def test_mesh_fits_model_test_job(mesh):
    degrees = mesh.model_dump(exclude={"expert_parallel"})
    assert all(degree > 0 for degree in degrees.values())
    assert prod(degrees.values()) == 8
    assert mesh.tensor_parallel == mesh.context_parallel_replicate == mesh.context_parallel_shard == 1
    dp_degree = mesh.data_parallel_replicate * mesh.data_parallel_shard
    # The parity helpers require DTensor parameters, and split a batch of 16 into two microbatches.
    assert dp_degree > 1
    assert 16 % (dp_degree * 2) == 0
    assert dp_degree % mesh.expert_parallel == 0
    assert 8 % mesh.expert_parallel == 0


@pytest.mark.parametrize("meshes", [DENSE_MODEL_MESHES, MOE_MODEL_MESHES], ids=["dense", "moe"])
def test_meshes_cover_data_parallelism_with_and_without_pipeline(meshes):
    actual = {
        (mesh.has_pipeline_parallel, mesh.has_data_parallel_replicate, mesh.has_data_parallel_shard)
        for mesh in meshes.values()
        if not mesh.has_expert_parallel
    }
    expected = {
        (pp, replicate, shard)
        for pp in (False, True)
        for replicate, shard in ((True, False), (False, True), (True, True))
    }
    assert actual == expected
    assert any(mesh.pipeline_parallel > 2 for mesh in meshes.values())
    assert len(meshes) == len(set(meshes.values()))


def test_expert_parallelism_is_only_tested_for_moe():
    assert all(not mesh.has_expert_parallel for mesh in DENSE_MODEL_MESHES.values())
    assert DENSE_MODEL_MESHES.items() <= MOE_MODEL_MESHES.items()


def test_moe_covers_expert_shards_with_and_without_replicas():
    actual = {
        (
            mesh.has_pipeline_parallel,
            mesh.data_parallel_replicate * mesh.data_parallel_shard // mesh.expert_parallel > 1,
        )
        for mesh in MOE_MODEL_MESHES.values()
        if mesh.has_expert_parallel
    }
    assert actual == set(product((False, True), repeat=2))

    hsdp_ep_meshes = [
        mesh
        for mesh in MOE_MODEL_MESHES.values()
        if mesh.has_data_parallel_replicate and mesh.has_data_parallel_shard and mesh.has_expert_parallel
    ]
    assert any(mesh.expert_parallel <= mesh.data_parallel_shard for mesh in hsdp_ep_meshes)
    assert any(mesh.expert_parallel > mesh.data_parallel_shard for mesh in hsdp_ep_meshes)
