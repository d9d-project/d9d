import os
import tempfile
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DeviceMeshDomain, DeviceMeshParameters, DistributedContext, init_dist_context


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(123)


@pytest.fixture(scope="session", autouse=True)
def default_process_group() -> Generator[None, None, None]:
    # The harness owns the default group, so that closing a cached context never destroys it.
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group()
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


DistContextFactory = Callable[..., DistributedContext]


@pytest.fixture(scope="session")
def dist_ctx_factory(default_process_group) -> Generator[DistContextFactory, None, None]:
    cache: dict[tuple[DeviceMeshParameters, tuple[DeviceMeshDomain, ...]], DistributedContext] = {}

    with ExitStack() as stack:

        def _get_context(
            params: DeviceMeshParameters, extra_domains: Sequence[DeviceMeshDomain] = ()
        ) -> DistributedContext:
            key = (params, tuple(extra_domains))
            if key not in cache:
                cache[key] = stack.enter_context(init_dist_context(params, extra_domains))
            return cache[key]

        yield _get_context


@pytest.fixture
def shared_tmp_dir(dist_ctx_factory) -> Generator[Path, None, None]:
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))
    gathered: list[Path | None] = [None] * dist_ctx.world_size
    if dist_ctx.is_main_process:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            dist.all_gather_object(gathered, tmp_dir)
            yield tmp_dir
            dist_ctx.wait_world()  # wait before destroying the directory
    else:
        dist.all_gather_object(gathered, None)
        tmp_dir = cast(Path, gathered[0])  # rank 0 sends the directory
        yield tmp_dir
        dist_ctx.wait_world()  # wait before destroying the directory
