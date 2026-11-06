from d9d.dist_context import DATA_SHARD_DIMS
from d9d.parallel.runtime import ParameterRuntime

from ..config import ParameterRuntimeConfig, dtype_from_name


def build_parameter_runtime(config: ParameterRuntimeConfig) -> ParameterRuntime:
    """Builds the ``ParameterRuntime`` of a job.

    Args:
        config: The runtime configuration.

    Returns:
        A runtime that gathers parameters along ``DATA_SHARD_DIMS``.
    """
    return ParameterRuntime(
        gather_dims=DATA_SHARD_DIMS,
        bucket_size_mib=config.bucket_size_mib,
        compute_dtype=None if config.compute_dtype is None else dtype_from_name(config.compute_dtype),
        memory_budget_bytes=config.memory_budget_mib * 1024 * 1024,
    )
