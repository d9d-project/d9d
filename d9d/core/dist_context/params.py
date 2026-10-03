import logging
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from .configured import DistributedContext


class DeviceMeshParameters(BaseModel):
    """Parallelism degrees to build the device meshes from.

    Attributes:
        pipeline_parallel: Degree of pipeline parallelism (PP).
        data_parallel_replicate: Degree of data-parallel replication (DDP).
        data_parallel_shard: Degree of data-parallel sharding (FSDP).
        context_parallel_replicate: Degree of context-parallel (CP) replication.
        context_parallel_shard: Degree of context-parallel (CP) sharding.
        tensor_parallel: Degree of tensor parallelism (TP).
        expert_parallel: Degree of expert parallelism (EP) for MoE layers.
    """

    model_config = ConfigDict(frozen=True)

    pipeline_parallel: int = 1

    data_parallel_replicate: int = 1
    data_parallel_shard: int = 1

    context_parallel_replicate: int = 1
    context_parallel_shard: int = 1

    tensor_parallel: int = 1

    expert_parallel: int = 1

    @property
    def has_pipeline_parallel(self) -> bool:
        """Whether pipeline parallelism is enabled (degree > 1)."""
        return self.pipeline_parallel > 1

    @property
    def has_data_parallel_replicate(self) -> bool:
        """Whether data parallel replication is enabled (degree > 1)."""
        return self.data_parallel_replicate > 1

    @property
    def has_data_parallel_shard(self) -> bool:
        """Whether data parallel sharding is enabled (degree > 1)."""
        return self.data_parallel_shard > 1

    @property
    def has_context_parallel_replicate(self) -> bool:
        """Whether context parallel replication is enabled (degree > 1)."""
        return self.context_parallel_replicate > 1

    @property
    def has_context_parallel_shard(self) -> bool:
        """Whether context parallel sharding is enabled (degree > 1)."""
        return self.context_parallel_shard > 1

    @property
    def has_tensor_parallel(self) -> bool:
        """Whether tensor parallelism is enabled (degree > 1)."""
        return self.tensor_parallel > 1

    @property
    def has_expert_parallel(self) -> bool:
        """Whether expert parallelism is enabled (degree > 1)."""
        return self.expert_parallel > 1

    @property
    def is_distributed(self) -> bool:
        """Whether any form of parallelism is enabled."""
        return (
            self.has_pipeline_parallel
            or self.has_data_parallel_replicate
            or self.has_data_parallel_shard
            or self.has_context_parallel_shard
            or self.has_context_parallel_replicate
            or self.has_expert_parallel
            or self.has_tensor_parallel
        )

    @model_validator(mode="after")
    def _check_ep_divisibility(self) -> Self:
        dp_cp_tp_degree = (
            self.data_parallel_shard
            * self.data_parallel_replicate
            * self.context_parallel_shard
            * self.context_parallel_replicate
            * self.tensor_parallel
        )
        ep_degree = self.expert_parallel

        if dp_cp_tp_degree % ep_degree != 0:
            raise ValueError(
                f"Total data/context/tensor parallelism degree ({dp_cp_tp_degree}) must be divisible by "
                f"total expert parallelism degree ({ep_degree})."
            )
        return self

    def build(self, log_level: int = logging.INFO) -> "DistributedContext":
        """Builds a ``DistributedContext`` from these parameters.

        Args:
            log_level: The log level of the ``d9d`` logger.

        Returns:
            A new ``DistributedContext`` with the built device meshes.
        """
        return DistributedContext(self, log_level)
