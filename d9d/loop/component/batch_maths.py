from d9d.core.dist_context import BATCH_DOMAIN, DistributedContext
from d9d.loop.config import BatchingConfig, PipeliningConfig


class BatchMaths:
    def __init__(
            self,
            dist_context: DistributedContext,
            config_batching: BatchingConfig,
            config_pipelining: PipeliningConfig | None
    ):
        self._dist_context = dist_context
        self._config_batching = config_batching
        self._config_pipelining = config_pipelining

        forward_batch_size = config_batching.microbatch_size * dist_context.mesh_for(BATCH_DOMAIN)["dp"].size()
        if config_pipelining is not None:
            forward_batch_size *= config_pipelining.n_sequential_microbatches

        if config_batching.global_batch_size % forward_batch_size != 0:
            raise ValueError(f"Global batch size {config_batching.global_batch_size} should be divisible "
                             f"by forward batch size {forward_batch_size}")

        self._forward_batch_size = forward_batch_size

    @property
    def data_loader_batch_size(self) -> int:
        if self._config_pipelining is None:
            return self._config_batching.microbatch_size
        else:
            return self._config_batching.microbatch_size * self._config_pipelining.n_sequential_microbatches

    @property
    def pipeline_schedule_n_microbatches(self) -> int:
        if self._config_pipelining is None:
            raise ValueError("Pipelining should be configured")
        return self._config_pipelining.n_sequential_microbatches

    @property
    def single_forward_batch_size(self) -> int:
        return self._forward_batch_size

    @property
    def sequential_forward_steps(self) -> int:  # used for gradient accumulation
        return self._config_batching.global_batch_size // self._forward_batch_size

    @property
    def gradient_will_be_accumulated_times(self) -> int:
        if self._config_pipelining is None:
            n_pipe = 1
        else:
            n_pipe = self._config_pipelining.n_sequential_microbatches

        n_grad = self.sequential_forward_steps

        return n_pipe * n_grad
