import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import cast

import pytest
import torch
from d9d.core.dist_context import REGULAR_DOMAIN, DeviceMeshParameters
from d9d.core.dist_ops import all_gather_object
from torch.distributed import GroupMember, destroy_process_group


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(123)


@pytest.fixture(scope="session", autouse=True)
def destroy_process_group_fixture():
    yield
    if GroupMember.WORLD is not None:
        destroy_process_group()


@pytest.fixture(scope="session")
def dist_ctx_dpr8():
    return DeviceMeshParameters(
        pipeline_parallel=1,
        tensor_parallel=1,
        expert_parallel=1,
        data_parallel_shard=1,
        context_parallel_shard=1,
        data_parallel_replicate=8,
        context_parallel_replicate=1
    ).build()


@pytest.fixture(scope="session")
def dist_ctx_pp():
    return DeviceMeshParameters(
        pipeline_parallel=8,
        context_parallel_shard=1,
        context_parallel_replicate=1,
        expert_parallel=1,
        tensor_parallel=1,
        data_parallel_shard=1,
        data_parallel_replicate=1
    ).build()


@pytest.fixture(scope="session")
def dist_ctx_pp4_dpr2():
    return DeviceMeshParameters(
        pipeline_parallel=4,
        context_parallel_shard=1,
        context_parallel_replicate=1,
        expert_parallel=1,
        tensor_parallel=1,
        data_parallel_shard=1,
        data_parallel_replicate=2
    ).build()


@pytest.fixture(scope="session")
def dist_ctx_local():
    return DeviceMeshParameters(
        pipeline_parallel=1,
        context_parallel_shard=1,
        context_parallel_replicate=1,
        expert_parallel=1,
        tensor_parallel=1,
        data_parallel_shard=1,
        data_parallel_replicate=1
    ).build()


@pytest.fixture
def shared_tmp_dir(dist_ctx_dpr8) -> Generator[Path, None, None]:
    reg_domain = dist_ctx_dpr8.mesh_for(REGULAR_DOMAIN)["dp_replicate"]
    if dist_ctx_dpr8.is_main_process:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            all_gather_object(tmp_dir, reg_domain.get_group())
            yield tmp_dir
    else:
        tmp_dir = next(x for x in all_gather_object(None, reg_domain.get_group()) if x is not None)
        tmp_dir = cast(Path, tmp_dir)  # we know it is received
        yield tmp_dir
