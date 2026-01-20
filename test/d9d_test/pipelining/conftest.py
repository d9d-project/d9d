import pytest
from d9d.core.dist_context import DeviceMeshParameters
from torch.distributed import destroy_process_group


@pytest.fixture(scope="session")
def dist_ctx_pp():
    ctx = DeviceMeshParameters(
        pipeline_parallel=8,
        context_parallel_shard=1,
        context_parallel_replicate=1,
        expert_parallel=1,
        tensor_parallel=1,
        data_parallel_shard=1,
        data_parallel_replicate=1
    ).build()
    yield ctx
    destroy_process_group(None)
