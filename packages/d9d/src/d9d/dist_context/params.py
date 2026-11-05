from typing import Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class DenseDomainConfig(BaseModel):
    """Configuration for the dense domain, which shards the parameters of dense layers.

    Attributes:
        data_parallel_shard: The number of data-parallel ranks that shard a parameter.
        context_parallel_shard: The number of context-parallel ranks that shard a parameter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_parallel_shard: PositiveInt = 1
    context_parallel_shard: PositiveInt = 1


class ExpertDomainConfig(BaseModel):
    """Configuration for the expert domain.

    Attributes:
        expert_parallel: The number of ranks that split the experts of a layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    expert_parallel: PositiveInt


class DeviceMeshParameters(BaseModel):
    """Configuration for the parallelism degrees of a job.

    Attributes:
        pipeline_parallel: The degree of pipeline parallelism.
        data_parallel: The degree of data parallelism.
        context_parallel: The degree of context parallelism.
        tensor_parallel: The degree of tensor parallelism.
        dense: The configuration of the dense domain.
        expert: The configuration of the expert domain. ``None`` means that the job builds no expert domain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline_parallel: PositiveInt = 1
    data_parallel: PositiveInt = 1
    context_parallel: PositiveInt = 1
    tensor_parallel: PositiveInt = 1
    dense: DenseDomainConfig = Field(default_factory=DenseDomainConfig)
    expert: ExpertDomainConfig | None = None

    @model_validator(mode="after")
    def _check_divisibility(self) -> Self:
        if self.data_parallel % self.dense.data_parallel_shard != 0:
            raise ValueError(
                f"The data-parallel shard size ({self.dense.data_parallel_shard}) must divide "
                f"the data-parallel degree ({self.data_parallel})."
            )
        if self.context_parallel % self.dense.context_parallel_shard != 0:
            raise ValueError(
                f"The context-parallel shard size ({self.dense.context_parallel_shard}) must divide "
                f"the context-parallel degree ({self.context_parallel})."
            )
        data_context_parallel = self.data_parallel * self.context_parallel
        if self.expert is not None and data_context_parallel % self.expert.expert_parallel != 0:
            raise ValueError(
                f"The expert-parallel degree ({self.expert.expert_parallel}) must divide "
                f"the product of the data-parallel and context-parallel degrees ({data_context_parallel})."
            )
        return self
