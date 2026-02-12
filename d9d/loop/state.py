import dataclasses
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from d9d.core.dist_context import DistributedContext
from d9d.core.protocol import LRSchedulerProtocol, OptimizerProtocol
from d9d.loop.component import (
    BatchMaths,
    GradientClipper,
    GradientManager,
    InferenceTaskOperator,
    JobLogger,
    JobProfiler,
    ManualGarbageCollector,
    ModelStageExporter,
    StateCheckpointer,
    Stepper,
    TimeoutManager,
    TrackedModules,
    TrainTaskOperator,
)
from d9d.loop.control import InferenceTask, TrainTask
from d9d.metric.impl import ComposeMetric


@dataclasses.dataclass(kw_only=True)
class JobState(Stateful):
    """
    Base container for the state of a distributed execution job.

    This dataclass holds the common infrastructure components required for both
    training and inference loops. It implements the Stateful protocol to support
    checkpointing of its internal components.

    Attributes:
        dist_context: The distributed context.
        stepper: Component for tracking the current global step and total steps.
        garbage_collector: Component for manual control of Python garbage collection.
        checkpointer: Component responsible for saving and loading execution states.
        profiler: Component for performance profiling.
        tracked_modules: Container holding the model (or model parts) being executed.
        batch_maths: Helper for calculating batch sizes and gradient accumulation steps.
        data_loader: The input data stream.
        timeout_manager: Component for checking and refreshing distributed timeouts.
    """

    dist_context: DistributedContext

    stepper: Stepper
    garbage_collector: ManualGarbageCollector
    checkpointer: StateCheckpointer
    profiler: JobProfiler

    tracked_modules: TrackedModules
    batch_maths: BatchMaths

    data_loader: StatefulDataLoader

    timeout_manager: TimeoutManager

    def state_dict(self) -> dict[str, Any]:
        return {
            "stepper": self.stepper.state_dict(),
            "tracked_modules": self.tracked_modules.state_dict(),
            "data_loader": self.data_loader.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.stepper.load_state_dict(state_dict["stepper"])
        self.tracked_modules.load_state_dict(state_dict["tracked_modules"])
        self.data_loader.load_state_dict(state_dict["data_loader"])


@dataclasses.dataclass(kw_only=True)
class TrainJobState(JobState):
    """
    Container for the state of a training job.

    Extends JobState to include components specific to training, such as
    optimization, gradient management, and loss computation.

    Attributes:
        task: The specific training task logic definition.
        gradient_manager: Component handling gradient synchronization.
        metrics: Container for aggregating training metrics.
        task_operator: Executor for running forward and backward passes.
        logger: Component for logging metrics and system status.
        optimizer: The optimizer instance updating model parameters.
        lr_scheduler: The scheduler adjusting the learning rate.
        gradient_clipper: Component for clipping gradient norms.
        exporter: Component for exporting the final model artifacts.
    """

    task: TrainTask
    gradient_manager: GradientManager
    metrics: ComposeMetric
    task_operator: TrainTaskOperator

    logger: JobLogger

    optimizer: OptimizerProtocol
    lr_scheduler: LRSchedulerProtocol
    gradient_clipper: GradientClipper
    exporter: ModelStageExporter

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "logger": self.logger.state_dict(),
            "task": self.task.state_dict(),
            "metrics": self.metrics.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)

        self.logger.load_state_dict(state_dict["logger"])
        self.task.load_state_dict(state_dict["task"])
        self.metrics.load_state_dict(state_dict["metrics"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


@dataclasses.dataclass(kw_only=True)
class InferenceJobState(JobState):
    """
    Container for the state of an inference job.

    Attributes:
        task: The specific inference task logic definition.
        task_operator: Executor for running forward and backward passes.
    """

    task: InferenceTask
    task_operator: InferenceTaskOperator

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "task": self.task.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.task.load_state_dict(state_dict["task"])
