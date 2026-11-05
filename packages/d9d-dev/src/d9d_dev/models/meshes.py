from d9d.dist_context import DenseDomainConfig, DeviceMeshParameters, ExpertDomainConfig

DENSE_MODEL_MESHES = {
    "ddp": DeviceMeshParameters(data_parallel=8),
    "fsdp": DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)),
    "hsdp": DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4)),
    "pp-ddp": DeviceMeshParameters(pipeline_parallel=4, data_parallel=2),
    "pp-fsdp": DeviceMeshParameters(
        pipeline_parallel=2, data_parallel=4, dense=DenseDomainConfig(data_parallel_shard=4)
    ),
    "pp-hsdp": DeviceMeshParameters(
        pipeline_parallel=2, data_parallel=4, dense=DenseDomainConfig(data_parallel_shard=2)
    ),
}

# A model with experts needs the expert section, so the dense layouts get experts without sharding.
MOE_MODEL_MESHES = {
    **{
        name: mesh.model_copy(update={"expert": ExpertDomainConfig(expert_parallel=1)})
        for name, mesh in DENSE_MODEL_MESHES.items()
    },
    "ddp-ep": DeviceMeshParameters(data_parallel=8, expert=ExpertDomainConfig(expert_parallel=8)),
    "fsdp-ep-replicated": DeviceMeshParameters(
        data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8), expert=ExpertDomainConfig(expert_parallel=2)
    ),
    "hsdp-ep-replicated": DeviceMeshParameters(
        data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4), expert=ExpertDomainConfig(expert_parallel=2)
    ),
    "hsdp-ep": DeviceMeshParameters(
        data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4), expert=ExpertDomainConfig(expert_parallel=8)
    ),
    "pp-fsdp-ep": DeviceMeshParameters(
        pipeline_parallel=2,
        data_parallel=4,
        dense=DenseDomainConfig(data_parallel_shard=4),
        expert=ExpertDomainConfig(expert_parallel=4),
    ),
    "pp-hsdp-ep-replicated": DeviceMeshParameters(
        pipeline_parallel=2,
        data_parallel=4,
        dense=DenseDomainConfig(data_parallel_shard=2),
        expert=ExpertDomainConfig(expert_parallel=2),
    ),
}
