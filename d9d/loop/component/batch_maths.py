from d9d.core.dist_context import BATCH_DOMAIN, DistributedContext
from d9d.loop.config import BatchingConfig, PipeliningConfig


class BatchMaths:
    """
    Calculates derived batching dimensions and iteration counts for distributed training loops.

    This class bridges the gap between global configuration (Global Batch Size) and
    local execution constraints (Microbatch Size, Data Parallel World Size).
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        config_batching: BatchingConfig,
        config_pipelining: PipeliningConfig | None,
    ):
        """
        Constructs the batch mathematics calculator.

        Validates that the Global Batch Size is perfectly divisible by the
        effective parallel microbatch capacity (DP size * Microbatch size).

        Args:
            dist_context: The distributed context containing mesh layout information.
            config_batching: Configuration detailing batch sizes.
            config_pipelining: Optional configuration for pipeline parallelism capabilities.

        Raises:
            ValueError: If global batch size is not divisible by the product of
                Data Parallel size and Microbatch size.
        """

        self._dist_context = dist_context
        self._config_batching = config_batching
        self._config_pipelining = config_pipelining

        global_batch = self._config_batching.global_batch_size
        if self._dist_context.mesh_params.is_distributed:
            dp_size = self._dist_context.mesh_for(BATCH_DOMAIN)["dp"].size()
        else:
            dp_size = 1
        microbatch_size = self._config_batching.microbatch_size

        global_microbatch = dp_size * microbatch_size

        if global_batch % global_microbatch != 0:
            raise ValueError("Global Batch Size must be divisible by (Data Parallel cardinality * Microbatch Size)")

        self._global_microbatch_size = global_microbatch

    @property
    def global_batch_size(self) -> int:
        """
        Returns the global batch size across the world.
        """

        return self._config_batching.global_batch_size

    @property
    def num_microbatches_pipelining(self) -> int:
        """
        Returns the number of microbatches handled by the pipeline scheduler per step.

        If pipeline parallelism is enabled, this is the total number of microbatches
        processed to form one global batch. If disabled, this returns 1.
        """

        if not self._dist_context.mesh_params.has_pipeline_parallel:
            return 1

        return self._config_batching.global_batch_size // self._global_microbatch_size

    @property
    def num_microbatches_gradient_accumulation(self) -> int:
        """
        Returns the number of gradient accumulation iterations for non-pipelined training.

        If pipeline parallelism is enabled, this returns 1 (as accumulation is handled
        internally by the pipeline schedule). If disabled, this is the number of
        forward/backward passes the training loop must execute before an optimizer step.
        """

        if self._dist_context.mesh_params.has_pipeline_parallel:
            return 1

        return self._config_batching.global_batch_size // self._global_microbatch_size

    @property
    def data_loader_batch_size(self) -> int:
        """
        Returns the quantity of samples this local rank needs to fetch for one optimizer step.

        This is calculated as `microbatch_size * total_microbatches_per_step`.
        """

        return self._config_batching.microbatch_size * self.num_microbatches_pipelining

    @property
    def num_backward_calls(self) -> int:
        """
        Returns the total number of backward passes executed per optimizer step.

        This represents the total gradient accumulation factor, regardless of whether
        it is handled by a pipeline schedule or a simple loop.
        """

        return self.num_microbatches_pipelining * self.num_microbatches_gradient_accumulation
