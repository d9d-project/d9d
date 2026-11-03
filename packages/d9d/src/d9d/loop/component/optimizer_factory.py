from d9d.dist_context import DistributedContext
from d9d.optim import LRSchedulerProtocol, OptimizerProtocol

from ..control import (
    InitializeLRSchedulerContext,
    InitializeOptimizerStageContext,
    LRSchedulerProvider,
    OptimizerProvider,
)
from .job_schedule import JobSchedule
from .model_stage_factory import TrackedModules
from .pipelined_lr_scheduler import PipelinedLRScheduler
from .pipelined_optimizer import PipelinedOptimizer


class OptimizerFactory:
    """Factory of the optimizer and the learning rate scheduler for the model stages of this rank."""

    def __init__(
        self,
        dist_context: DistributedContext,
        tracked_modules: TrackedModules,
        optimizer_provider: OptimizerProvider,
        lr_scheduler_provider: LRSchedulerProvider,
        schedule: JobSchedule,
    ):
        """Constructs the ``OptimizerFactory`` object.

        Args:
            dist_context: The distributed context.
            tracked_modules: The model stages owned by the current rank.
            optimizer_provider: The provider that creates an optimizer for one model stage.
            lr_scheduler_provider: The provider that creates an LR scheduler for one optimizer.
            schedule: The job schedule that provides the total number of steps.
        """
        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._optimizer_provider = optimizer_provider
        self._lr_scheduler_provider = lr_scheduler_provider
        self._schedule = schedule

    def build_optimizer_and_scheduler(self) -> tuple[OptimizerProtocol, LRSchedulerProtocol]:
        """Builds the optimizer and the learning rate scheduler.

        The providers create one optimizer and one scheduler per local model stage. They are combined
        into a ``PipelinedOptimizer`` and a ``PipelinedLRScheduler`` that step all stages together.

        Returns:
            A tuple of the pipeline-aware optimizer and scheduler.
        """
        optimizers: dict[int, OptimizerProtocol] = {}
        lr_schedulers: dict[int, LRSchedulerProtocol] = {}
        for stage_index, module in self._tracked_modules.modules.items():
            optimizer = self._optimizer_provider(
                InitializeOptimizerStageContext(dist_context=self._dist_context, model=module)
            )
            optimizers[stage_index] = optimizer

            scheduler = self._lr_scheduler_provider(
                InitializeLRSchedulerContext(
                    dist_context=self._dist_context, total_steps=self._schedule.total_steps, optimizer=optimizer
                )
            )
            lr_schedulers[stage_index] = scheduler

        pipe_optimizer = PipelinedOptimizer(optimizers=optimizers)
        pipe_scheduler = PipelinedLRScheduler(schedulers=lr_schedulers)
        return pipe_optimizer, pipe_scheduler
