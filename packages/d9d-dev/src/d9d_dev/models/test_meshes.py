from itertools import product

import pytest

from d9d_dev.models.meshes import DENSE_MODEL_MESHES, MOE_MODEL_MESHES

pytestmark = pytest.mark.local


def _expert_parallel(mesh) -> int:
    return 1 if mesh.expert is None else mesh.expert.expert_parallel


@pytest.mark.parametrize(
    "mesh",
    [
        pytest.param(mesh, id=f"{model_kind}-{name}")
        for model_kind, meshes in (("dense", DENSE_MODEL_MESHES), ("moe", MOE_MODEL_MESHES))
        for name, mesh in meshes.items()
    ],
)
def test_mesh_fits_model_test_job(mesh):
    assert mesh.pipeline_parallel * mesh.data_parallel * mesh.context_parallel * mesh.tensor_parallel == 8
    assert mesh.tensor_parallel == mesh.context_parallel == 1
    # The parity helpers require DTensor parameters, and split a batch of 16 into two microbatches.
    assert mesh.data_parallel > 1
    assert 16 % (mesh.data_parallel * 2) == 0
    assert mesh.data_parallel % _expert_parallel(mesh) == 0


@pytest.mark.parametrize("meshes", [DENSE_MODEL_MESHES, MOE_MODEL_MESHES], ids=["dense", "moe"])
def test_meshes_cover_data_parallelism_with_and_without_pipeline(meshes):
    actual = {
        (
            mesh.pipeline_parallel > 1,
            mesh.data_parallel // mesh.dense.data_parallel_shard > 1,
            mesh.dense.data_parallel_shard > 1,
        )
        for mesh in meshes.values()
        if _expert_parallel(mesh) == 1
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
    assert all(mesh.expert is None for mesh in DENSE_MODEL_MESHES.values())
    assert all(mesh.expert is not None for mesh in MOE_MODEL_MESHES.values())
    assert DENSE_MODEL_MESHES.keys() <= MOE_MODEL_MESHES.keys()


def test_moe_covers_expert_shards_with_and_without_replicas():
    actual = {
        (mesh.pipeline_parallel > 1, mesh.data_parallel // _expert_parallel(mesh) > 1)
        for mesh in MOE_MODEL_MESHES.values()
        if _expert_parallel(mesh) > 1
    }
    assert actual == set(product((False, True), repeat=2))

    hsdp_ep_meshes = [
        mesh
        for mesh in MOE_MODEL_MESHES.values()
        if 1 < mesh.dense.data_parallel_shard < mesh.data_parallel and _expert_parallel(mesh) > 1
    ]
    assert any(_expert_parallel(mesh) <= mesh.dense.data_parallel_shard for mesh in hsdp_ep_meshes)
    assert any(_expert_parallel(mesh) > mesh.dense.data_parallel_shard for mesh in hsdp_ep_meshes)
