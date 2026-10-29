from pathlib import Path

import pytest
import torch
from d9d.core.dist_context import (
    BATCH_DOMAIN,
    DENSE_DOMAIN,
    EXPERT_DOMAIN,
    FLAT_DOMAIN,
    REGULAR_DOMAIN,
    DeviceMeshParameters,
)
from pydantic import ValidationError


@pytest.mark.local
@pytest.mark.parametrize(
    "params_dict",
    [
        {"pp": 1, "dpr": 1, "dps": 1, "cpr": 1, "cps": 1, "tp": 1, "ep": 1},
        {"pp": 4, "dpr": 1, "dps": 1, "cpr": 1, "cps": 1, "tp": 1, "ep": 1},
        {"pp": 1, "dpr": 4, "dps": 1, "cpr": 1, "cps": 1, "tp": 1, "ep": 1},
        {"pp": 1, "dpr": 1, "dps": 4, "cpr": 1, "cps": 1, "tp": 1, "ep": 1},
        {"pp": 1, "dpr": 1, "dps": 1, "cpr": 4, "cps": 1, "tp": 1, "ep": 1},
        {"pp": 1, "dpr": 1, "dps": 1, "cpr": 1, "cps": 4, "tp": 1, "ep": 1},
        {"pp": 1, "dpr": 1, "dps": 1, "cpr": 1, "cps": 1, "tp": 4, "ep": 1},
        {"pp": 1, "dpr": 4, "dps": 1, "cpr": 1, "cps": 1, "tp": 1, "ep": 4},
        {"pp": 1, "dpr": 1, "dps": 4, "cpr": 1, "cps": 1, "tp": 1, "ep": 4},
        {"pp": 1, "dpr": 1, "dps": 1, "cpr": 4, "cps": 1, "tp": 1, "ep": 4},
        {"pp": 1, "dpr": 1, "dps": 1, "cpr": 1, "cps": 4, "tp": 1, "ep": 4},
        {"pp": 1, "dpr": 1, "dps": 1, "cpr": 1, "cps": 1, "tp": 4, "ep": 4},
        {"pp": 1, "dpr": 1, "dps": 2, "cpr": 1, "cps": 1, "tp": 2, "ep": 4},
        {"pp": 1, "dpr": 8, "dps": 2, "cpr": 1, "cps": 1, "tp": 1, "ep": 4},
    ],
)
def test_params_validation_valid(params_dict):
    params = DeviceMeshParameters(
        pipeline_parallel=params_dict["pp"],
        data_parallel_replicate=params_dict["dpr"],
        data_parallel_shard=params_dict["dps"],
        context_parallel_replicate=params_dict["cpr"],
        context_parallel_shard=params_dict["cps"],
        tensor_parallel=params_dict["tp"],
        expert_parallel=params_dict["ep"],
    )

    # If any dimension > 1, it should be distributed
    expected_distributed = any(v > 1 for v in params_dict.values())
    assert params.is_distributed == expected_distributed

    # Verify specific flags based on input
    assert params.has_pipeline_parallel == (params_dict["pp"] > 1)
    assert params.has_data_parallel_replicate == (params_dict["dpr"] > 1)
    assert params.has_data_parallel_shard == (params_dict["dps"] > 1)
    assert params.has_context_parallel_replicate == (params_dict["cpr"] > 1)
    assert params.has_context_parallel_shard == (params_dict["cps"] > 1)
    assert params.has_tensor_parallel == (params_dict["tp"] > 1)
    assert params.has_expert_parallel == (params_dict["ep"] > 1)


@pytest.mark.local
def test_params_validation_invalid_ep():
    with pytest.raises(ValidationError):
        DeviceMeshParameters(
            pipeline_parallel=1,
            data_parallel_shard=4,
            data_parallel_replicate=1,
            context_parallel_shard=1,
            context_parallel_replicate=1,
            tensor_parallel=2,
            expert_parallel=3,
        )


@pytest.mark.local
def test_local_context_initialization():
    params = DeviceMeshParameters(
        pipeline_parallel=1,
        data_parallel_replicate=1,
        data_parallel_shard=1,
        context_parallel_replicate=1,
        context_parallel_shard=1,
        tensor_parallel=1,
        expert_parallel=1,
    )

    assert not params.is_distributed

    ctx = params.build()
    assert ctx.num_nodes == 1
    assert ctx.is_local_main_process
    assert ctx.is_main_process
    assert ctx.master_addr == "127.0.0.1"
    # Local context has no meshes
    with pytest.raises(ValueError):
        ctx.mesh_for(REGULAR_DOMAIN)


@pytest.mark.distributed
def test_mesh_topology_complex(dist_ctx_factory):
    ctx = dist_ctx_factory(
        DeviceMeshParameters(
            data_parallel_replicate=2,
            data_parallel_shard=2,
            tensor_parallel=2,
            expert_parallel=2,
        )
    )

    regular_mesh = ctx.mesh_for(REGULAR_DOMAIN)
    assert regular_mesh.ndim == 6
    assert regular_mesh.mesh_dim_names == ("pp", "dp_replicate", "dp_shard", "cp_shard", "cp_replicate", "tp")

    expert_mesh = ctx.mesh_for(EXPERT_DOMAIN)
    assert expert_mesh.ndim == 3
    assert expert_mesh.mesh_dim_names == ("pp", "ep_replicate", "ep_shard")

    dense_mesh = ctx.mesh_for(DENSE_DOMAIN)
    assert dense_mesh.ndim == 5
    assert dense_mesh.mesh_dim_names == ("pp", "dp_replicate", "dp_cp_shard", "cp_replicate", "tp")

    batch_mesh = ctx.mesh_for(BATCH_DOMAIN)
    assert batch_mesh.ndim == 4
    assert batch_mesh.mesh_dim_names == ("pp", "dp", "cp", "tp")

    flat_mesh = ctx.mesh_for(FLAT_DOMAIN)
    assert flat_mesh.ndim == 1
    assert flat_mesh.mesh_dim_names == ("world",)


@pytest.mark.distributed
def test_sync_primitives(dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_shard=8))

    # If this hangs, the test fails (Pytest timeout usually kills it)
    ctx.wait_world()

    ctx.set_timeout(10.0)  # Set to 10 seconds
    ctx.wait_world()


@pytest.mark.local
def test_sync_primitives_offline(dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters())

    # both wait and timeout shall not crash
    ctx.wait_world()

    ctx.set_timeout(10.0)
    ctx.wait_world()


@pytest.mark.distributed
def test_rank_assertions(dist_ctx_factory, shared_tmp_dir: Path):
    ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_shard=8))

    assert ctx.master_addr is not None
    assert len(ctx.master_addr) > 0

    assert ctx.num_nodes == 1

    if torch.distributed.get_rank() == 0:
        assert ctx.is_main_process
        assert ctx.is_local_main_process
    else:
        assert not ctx.is_main_process
        assert not ctx.is_local_main_process

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
