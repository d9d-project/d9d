from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.core.protocol import LRSchedulerProtocol, OptimizerProtocol
from d9d.loop.control import (
    InitializeLRSchedulerContext,
    InitializeOptimizerStageContext,
    LRSchedulerProvider,
    OptimizerProvider,
)
from d9d.pipelining.training import PipelinedLRScheduler, PipelinedOptimizer

from .job_schedule import JobSchedule
from .model_stage_factory import TrackedModules


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
        optimizers: list[OptimizerProtocol] = []
        lr_schedulers: list[LRSchedulerProtocol] = []
        for module in self._tracked_modules.modules:
            optimizer = self._optimizer_provider(
                InitializeOptimizerStageContext(dist_context=self._dist_context, model=module)
            )
            optimizers.append(optimizer)

            scheduler = self._lr_scheduler_provider(
                InitializeLRSchedulerContext(
                    dist_context=self._dist_context, total_steps=self._schedule.total_steps, optimizer=optimizer
                )
            )
            lr_schedulers.append(scheduler)

        if self._dist_context.mesh_params.is_distributed:
            mesh_pp = self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"]
        else:
            mesh_pp = None

        pipe_optimizer = PipelinedOptimizer(mesh_pp=mesh_pp, optimizers=optimizers)
        pipe_scheduler = PipelinedLRScheduler(mesh_pp=mesh_pp, schedulers=lr_schedulers)
        return pipe_optimizer, pipe_scheduler
