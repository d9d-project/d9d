from enum import StrEnum

from .domain import DeviceMeshDomain, MeshDim
from .params import DeviceMeshParameters


class BaseDim(StrEnum):
    """The dimensions of the mesh of ``BASE_DOMAIN``.

    Attributes:
        PP: The pipeline stages.
        DP: The data-parallel ranks, which take different samples.
        CP: The context-parallel ranks, which take different parts of one sample.
        TP: The tensor-parallel ranks, which split the matrix multiplications of one sample.
    """

    PP = "pp"
    DP = "dp"
    CP = "cp"
    TP = "tp"


class DenseDim(StrEnum):
    """The dimensions of the mesh of ``DENSE_DOMAIN``.

    Attributes:
        PP: The pipeline stages.
        DP_REPLICATE: The data-parallel ranks that hold the same shards of the dense parameters.
        DP_CP_SHARD: The data-parallel and context-parallel ranks that shard the dense parameters.
        CP_REPLICATE: The context-parallel ranks that hold the same shards of the dense parameters.
        TP: The tensor-parallel ranks.
    """

    PP = "pp"
    DP_REPLICATE = "dp_replicate"
    DP_CP_SHARD = "dp_cp_shard"
    CP_REPLICATE = "cp_replicate"
    TP = "tp"


class ExpertDim(StrEnum):
    """The dimensions of the mesh of ``EXPERT_DOMAIN``.

    Attributes:
        PP: The pipeline stages.
        EP_REPLICATE: The data-parallel and context-parallel ranks that hold the same experts.
        EP_SHARD: The data-parallel and context-parallel ranks that split the experts of a layer.
        TP: The tensor-parallel ranks.
    """

    PP = "pp"
    EP_REPLICATE = "ep_replicate"
    EP_SHARD = "ep_shard"
    TP = "tp"


class MetricDim(StrEnum):
    """The dimensions of the mesh of ``METRIC_DOMAIN``.

    Attributes:
        PP_DP_CP: All ranks with their own tokens or pipeline stage.
        TP: The tensor-parallel ranks, which share their tokens.
    """

    PP_DP_CP = "pp_dp_cp"
    TP = "tp"


class _BaseDomain(DeviceMeshDomain[BaseDim]):
    @property
    def name(self) -> str:
        return "base"

    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[BaseDim], ...]:
        return (
            MeshDim(BaseDim.PP, params.pipeline_parallel),
            MeshDim(BaseDim.DP, params.data_parallel),
            MeshDim(BaseDim.CP, params.context_parallel),
            MeshDim(BaseDim.TP, params.tensor_parallel),
        )


class _DenseDomain(DeviceMeshDomain[DenseDim]):
    @property
    def name(self) -> str:
        return "dense"

    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[DenseDim], ...]:
        return (
            MeshDim(DenseDim.PP, params.pipeline_parallel),
            MeshDim(DenseDim.DP_REPLICATE, params.data_parallel // params.dense.data_parallel_shard),
            MeshDim(DenseDim.DP_CP_SHARD, params.dense.data_parallel_shard * params.dense.context_parallel_shard),
            MeshDim(DenseDim.CP_REPLICATE, params.context_parallel // params.dense.context_parallel_shard),
            MeshDim(DenseDim.TP, params.tensor_parallel),
        )


class _ExpertDomain(DeviceMeshDomain[ExpertDim]):
    @property
    def name(self) -> str:
        return "expert"

    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[ExpertDim], ...]:
        if params.expert is None:
            raise ValueError("The expert domain needs the expert section of the mesh configuration. Set mesh.expert.")
        expert_parallel = params.expert.expert_parallel
        return (
            MeshDim(ExpertDim.PP, params.pipeline_parallel),
            MeshDim(ExpertDim.EP_REPLICATE, params.data_parallel * params.context_parallel // expert_parallel),
            MeshDim(ExpertDim.EP_SHARD, expert_parallel),
            MeshDim(ExpertDim.TP, params.tensor_parallel),
        )


class _MetricDomain(DeviceMeshDomain[MetricDim]):
    @property
    def name(self) -> str:
        return "metric"

    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[MetricDim], ...]:
        return (
            MeshDim(MetricDim.PP_DP_CP, params.pipeline_parallel * params.data_parallel * params.context_parallel),
            MeshDim(MetricDim.TP, params.tensor_parallel),
        )


BASE_DOMAIN = _BaseDomain()
"""The domain of the four parallelism degrees. The order of its dimensions defines how the ranks lie in the world."""

DENSE_DOMAIN = _DenseDomain()
"""The domain of the dense layers, which splits the data-parallel and context-parallel ranks into replicas and
shards."""

EXPERT_DOMAIN = _ExpertDomain()
"""The domain of the expert layers. The context builds it only if the expert section of the configuration is set."""

METRIC_DOMAIN = _MetricDomain()
"""The domain that sums the metrics and the loss weights, so that the tensor-parallel ranks count once."""


def _builtin_domains(params: DeviceMeshParameters) -> list[DeviceMeshDomain]:
    """Returns the built-in domains that a context with these parameters builds.

    Args:
        params: The parallelism degrees and the sections of the built-in domains.

    Returns:
        The built-in domains, with ``EXPERT_DOMAIN`` only if the expert section is set.
    """
    domains: list[DeviceMeshDomain] = [BASE_DOMAIN, DENSE_DOMAIN, METRIC_DOMAIN]
    if params.expert is not None:
        domains.append(EXPERT_DOMAIN)
    return domains
