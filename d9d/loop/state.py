import dataclasses
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext
from d9d.core.protocol import LRSchedulerProtocol, MicrobatchPackStream, OptimizerProtocol
from d9d.loop.component import (
    GradientClipper,
    GradientManager,
    InferenceTaskOperator,
    JobLogger,
    JobProfiler,
    JobSchedule,
    ManualGarbageCollector,
    ModelStageExporter,
    StateCheckpointer,
    TimeoutManager,
    TrackedModules,
    TrainTaskOperator,
)
from d9d.loop.control import InferenceTask, TrainTask
from d9d.loop.event import EventBus
from d9d.metric.impl.container import ComposeMetric


@dataclasses.dataclass(kw_only=True)
class JobState(Stateful):
    """Base container for the state of a distributed execution job.

    This dataclass holds the common infrastructure components required for both
    training and inference loops. It implements the Stateful protocol to support
    checkpointing of its internal components.

    Attributes:
        dist_context: The distributed context.
        schedule: Component for tracking the current global step and total steps.
        garbage_collector: Component for manual control of Python garbage collection.
        checkpointer: Component responsible for saving and loading execution states.
        profiler: Component for performance profiling.
        tracked_modules: Container holding the model (or model parts) being executed.
        microbatch_pack_stream: The microbatch pack stream feeding the loop.
        timeout_manager: Component for checking and refreshing distributed timeouts.
    """

    dist_context: DistributedContext

    schedule: JobSchedule
    garbage_collector: ManualGarbageCollector
    checkpointer: StateCheckpointer
    profiler: JobProfiler

    tracked_modules: TrackedModules

    microbatch_pack_stream: MicrobatchPackStream

    timeout_manager: TimeoutManager

    def state_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.state_dict(),
            "tracked_modules": self.tracked_modules.state_dict(),
            "microbatch_pack_stream": self.microbatch_pack_stream.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.schedule.load_state_dict(state_dict["schedule"])
        self.tracked_modules.load_state_dict(state_dict["tracked_modules"])
        self.microbatch_pack_stream.load_state_dict(state_dict["microbatch_pack_stream"])


@dataclasses.dataclass(kw_only=True)
class TrainJobState(JobState):
    """Container for the state of a training job.

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
    event_bus: EventBus

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
    """Container for the state of an inference job.

    Attributes:
        task: The specific inference task logic definition.
        task_operator: Executor for running forward and backward passes.
        event_bus: The event bus for this inference job.
    """

    task: InferenceTask
    task_operator: InferenceTaskOperator
    event_bus: EventBus

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "task": self.task.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.task.load_state_dict(state_dict["task"])
