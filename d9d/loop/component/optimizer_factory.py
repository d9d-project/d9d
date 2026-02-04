from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.core.protocol import LRSchedulerProtocol, OptimizerProtocol
from d9d.loop.control import (
    InitializeLRSchedulerContext,
    InitializeOptimizerStageContext,
    LRSchedulerProvider,
    OptimizerProvider,
)
from d9d.pipelining.training import PipelinedLRScheduler, PipelinedOptimizer

from .model_stage_factory import TrackedModules
from .stepper import Stepper


class OptimizerFactory:
    """
    Factory for creating and configuring distributed optimizers and learning rate schedulers.

    This factory handles the orchestration of optimizer creation for models potentially split across
    pipeline stages. It uses the providers to instantiate underlying PyTorch optimizers and schedulers for each
    tracked module, and wraps them in pipeline-aware interfaces.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            tracked_modules: TrackedModules,
            optimizer_provider: OptimizerProvider,
            lr_scheduler_provider: LRSchedulerProvider,
            stepper: Stepper
    ):
        """
        Constructs the OptimizerFactory.

        Args:
            dist_context: The distributed context.
            tracked_modules: A container of model modules owned by the current rank.
            optimizer_provider: A callable responsible for creating optimizer instances for a given model.
            lr_scheduler_provider: A callable responsible for creating LR scheduler instances.
            stepper: The training stepper providing information about total training steps.
        """
        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._optimizer_provider = optimizer_provider
        self._lr_scheduler_provider = lr_scheduler_provider
        self._stepper = stepper

    def build_optimizer_and_scheduler(self) -> tuple[OptimizerProtocol, LRSchedulerProtocol]:
        """
        Builds both the optimizer and learning rate scheduler.

        This method iterates through all local model modules. For each module, it creates an
        optimizer and scheduler using the configured providers. Finally, it aggregates these individual
        instances into a single `PipelinedOptimizer` and `PipelinedLRScheduler` capable of coordinated
        stepping across the pipeline parallel dimension.

        Returns:
            A tuple containing the initialized pipeline-aware optimizer and scheduler.
        """

        optimizers: list[OptimizerProtocol] = []
        lr_schedulers: list[LRSchedulerProtocol] = []
        for module in self._tracked_modules.modules:
            optimizer = self._optimizer_provider(
                InitializeOptimizerStageContext(
                    dist_context=self._dist_context,
                    model=module
                )
            )
            optimizers.append(optimizer)

            scheduler = self._lr_scheduler_provider(
                InitializeLRSchedulerContext(
                    dist_context=self._dist_context,
                    total_steps=self._stepper.total_steps,
                    optimizer=optimizer
                )
            )
            lr_schedulers.append(scheduler)
        pipe_optimizer = PipelinedOptimizer(
            mesh_pp=self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"],
            optimizers=optimizers
        )
        pipe_scheduler = PipelinedLRScheduler(
            mesh_pp=self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"],
            schedulers=lr_schedulers
        )
        return pipe_optimizer, pipe_scheduler
