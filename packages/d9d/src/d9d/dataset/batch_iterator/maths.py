from d9d.core.dist_context import BATCH_DOMAIN, DistributedContext


def num_microbatches_for_global_batch(
    dist_context: DistributedContext, global_batch_size: int, microbatch_size: int
) -> int:
    """Computes the number of microbatches per step required to reach a target global batch size.

    The global batch is spread across the data-parallel ranks, and each rank processes ``microbatch_size`` samples
    per microbatch. So each rank processes ``global_batch_size / (dp_size * microbatch_size)`` microbatches per
    optimizer step.

    Args:
        dist_context: The distributed context.
        global_batch_size: The total effective batch size across all replicas and accumulation.
        microbatch_size: The number of samples in a single microbatch on a single rank.

    Returns:
        The number of microbatches per step (the gradient-accumulation factor).

    Raises:
        ValueError: If ``global_batch_size`` is not divisible by ``dp_size * microbatch_size``.
    """
    if dist_context.mesh_params.is_distributed:
        dp_size = dist_context.mesh_for(BATCH_DOMAIN)["dp"].size()
    else:
        dp_size = 1

    global_microbatch = dp_size * microbatch_size

    if global_batch_size % global_microbatch != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"dp_size * microbatch_size ({dp_size} * {microbatch_size} = {global_microbatch})."
        )

    return global_batch_size // global_microbatch
