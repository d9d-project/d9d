from d9d.core.dist_context import DeviceMeshParameters

DENSE_MODEL_MESHES = {
    "ddp": DeviceMeshParameters(data_parallel_replicate=8),
    "fsdp": DeviceMeshParameters(data_parallel_shard=8),
    "hsdp": DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4),
    "pp-ddp": DeviceMeshParameters(pipeline_parallel=4, data_parallel_replicate=2),
    "pp-fsdp": DeviceMeshParameters(pipeline_parallel=2, data_parallel_shard=4),
    "pp-hsdp": DeviceMeshParameters(pipeline_parallel=2, data_parallel_replicate=2, data_parallel_shard=2),
}

MOE_MODEL_MESHES = {
    **DENSE_MODEL_MESHES,
    "ddp-ep": DeviceMeshParameters(data_parallel_replicate=8, expert_parallel=8),
    "fsdp-ep-replicated": DeviceMeshParameters(data_parallel_shard=8, expert_parallel=2),
    "hsdp-ep-replicated": DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4, expert_parallel=2),
    "hsdp-ep": DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4, expert_parallel=8),
    "pp-fsdp-ep": DeviceMeshParameters(pipeline_parallel=2, data_parallel_shard=4, expert_parallel=4),
    "pp-hsdp-ep-replicated": DeviceMeshParameters(
        pipeline_parallel=2, data_parallel_replicate=2, data_parallel_shard=2, expert_parallel=2
    ),
}
