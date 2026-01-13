from typing import Self

from pydantic import BaseModel, model_validator

from .configured import DistributedContext


class DeviceMeshParameters(BaseModel):
    """
    Configuration parameters for initializing Distributed Device Meshes.

    Attributes:
        pipeline_parallel: Degree of pipeline parallelism (PP).
        data_parallel_replicate: Degree of data parallel replication (DDP).
        data_parallel_shard: Degree of data parallel sharding (FSDP).
        context_parallel_replicate: Degree of context parallel (CP) replication.
        context_parallel_shard: Degree of context parallel (FSCP) sharding.
        tensor_parallel: Degree of tensor parallelism (TP).
        expert_parallel: Degree of expert parallelism (EP/MoE).
    """

    pipeline_parallel: int

    data_parallel_replicate: int
    data_parallel_shard: int

    context_parallel_replicate: int
    context_parallel_shard: int

    tensor_parallel: int

    expert_parallel: int

    @property
    def has_pipeline_parallel(self) -> bool:
        """Checks if pipeline parallelism is enabled (degree > 1)."""

        return self.pipeline_parallel > 1

    @property
    def has_data_parallel_replicate(self) -> bool:
        """Checks if data parallel replication is enabled (degree > 1)."""

        return self.data_parallel_replicate > 1

    @property
    def has_data_parallel_shard(self) -> bool:
        """Checks if data parallel sharding is enabled (degree > 1)."""

        return self.data_parallel_shard > 1

    @property
    def has_context_parallel_replicate(self) -> bool:
        return self.context_parallel_replicate > 1

    @property
    def has_context_parallel_shard(self) -> bool:
        return self.context_parallel_shard > 1

    @property
    def has_tensor_parallel(self) -> bool:
        return self.tensor_parallel > 1

    @property
    def has_expert_parallel(self) -> bool:
        """Checks if expert parallelism is enabled (degree > 1)."""
        return self.expert_parallel > 1

    @property
    def is_distributed(self) -> bool:
        """Checks if any form of parallelism is enabled."""

        return (
                self.has_pipeline_parallel or
                self.has_data_parallel_replicate or
                self.has_data_parallel_shard or
                self.has_context_parallel_shard or
                self.has_context_parallel_replicate or
                self.has_expert_parallel or
                self.has_tensor_parallel
        )

    @model_validator(mode='after')
    def _check_ep_divisibility(self) -> Self:
        """Validates that DP/CP/TP dimensions can support the requested EP/ETP degrees."""
        dp_cp_tp_degree = (
            self.data_parallel_shard *
            self.data_parallel_replicate *
            self.context_parallel_shard *
            self.context_parallel_replicate *
            self.tensor_parallel
        )
        ep_degree = self.expert_parallel

        if dp_cp_tp_degree % ep_degree != 0:
            raise ValueError(
                f"Total data/context/tensor parallelism degree ({dp_cp_tp_degree}) must be divisible by "
                f"total expert parallelism degree ({ep_degree})."
            )
        return self

    def build(self) -> 'DistributedContext':
        """
        Initializes the DistributedContext using these parameters.

        Returns:
            A new DistributedContext instance containing the initialized device meshes.
        """

        return DistributedContext(self)
