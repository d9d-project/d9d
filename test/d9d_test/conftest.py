import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast

import pytest
import torch
from d9d.core.dist_context import REGULAR_DOMAIN, DeviceMeshParameters, DistributedContext
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
def dist_ctx_factory() -> Callable[[DeviceMeshParameters], DistributedContext]:
    cache: dict[DeviceMeshParameters, DistributedContext] = {}

    def _get_context(params: DeviceMeshParameters) -> DistributedContext:
        if params not in cache:
            cache[params] = params.build()
        return cache[params]

    return _get_context


@pytest.fixture
def shared_tmp_dir(dist_ctx_factory) -> Generator[Path, None, None]:
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    reg_domain = dist_ctx.mesh_for(REGULAR_DOMAIN)["dp_replicate"]
    if dist_ctx.is_main_process:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            all_gather_object(tmp_dir, reg_domain.get_group())
            yield tmp_dir
    else:
        tmp_dir = next(x for x in all_gather_object(None, reg_domain.get_group()) if x is not None)
        tmp_dir = cast(Path, tmp_dir)  # we know it is received
        yield tmp_dir
