from d9d.core.dist_context import DeviceMeshParameters

MESHES_FOR_MODEL_TESTS = [
    # DPR
    DeviceMeshParameters(
        data_parallel_replicate=8,
    ),
    # DPS
    DeviceMeshParameters(
        data_parallel_shard=8,
    ),
    # DPR + DPS
    DeviceMeshParameters(
        data_parallel_shard=4,
        data_parallel_replicate=2,
    ),
    # DPR / EP
    DeviceMeshParameters(
        expert_parallel=2,
        data_parallel_replicate=8,
    ),
    # DPS / EP
    DeviceMeshParameters(
        expert_parallel=2,
        data_parallel_shard=8,
    ),
    # DPR + DPS / EP
    DeviceMeshParameters(
        expert_parallel=2,
        data_parallel_shard=4,
        data_parallel_replicate=2,
    ),
    # PP + DPR / EP
    DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=4,
        data_parallel_replicate=4,
    ),
    # PP + DPS / EP
    DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=4,
        data_parallel_shard=4,
    ),
    # PP + DPR + DPS / EP
    DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=4,
        data_parallel_shard=2,
        data_parallel_replicate=2,
    ),
    # PP + DPR + DPS / EP + R
    DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=2,
        data_parallel_shard=2,
        data_parallel_replicate=2,
    ),
]
