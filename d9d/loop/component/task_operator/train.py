import torch

from d9d.core.dist_context import DistributedContext
from d9d.core.types import MicrobatchPack
from d9d.loop.control import ComputeLossContext, TrainTask, UpdateMetricsContext
from d9d.metric.impl.container import ComposeMetric
from d9d.pipelining.factory.factory import PipelineScheduleInfo

from ..gradient_manager import GradientManager
from ..job_schedule import JobSchedule
from ..pipeline_state import PipelineStateHandler
from .common import build_pipeline_microbatch_inputs


class LossComputer:
    """Computes and accumulates the training loss for each microbatch of a step.

    This component bridges the raw outputs of the model pipeline and the user-defined training task.
    """

    def __init__(
        self,
        state: PipelineStateHandler,
        task: TrainTask,
        schedule: JobSchedule,
        gradient_manager: GradientManager,
        metrics: ComposeMetric,
    ):
        """Constructs a new LossComputer.

        Args:
            state: Handler for the per-microbatch pipeline state.
            task: The user-defined training task containing loss computation logic.
            schedule: Component tracking current step and progress.
            gradient_manager: Accumulator of loss/weight for gradient reduction.
            metrics: Metric collection updated per microbatch.
        """
        self._state = state
        self._task = task
        self._schedule = schedule
        self._gradient_manager = gradient_manager
        self._metrics = metrics

    def __call__(self, pipeline_outputs: dict[str, torch.Tensor], microbatch_idx: int) -> torch.Tensor:
        """Computes the weighted loss for a microbatch and accumulates its loss/weight and metrics.

        Args:
            pipeline_outputs: Dictionary containing model output tensors.
            microbatch_idx: Index of the current microbatch within the step's pack.

        Returns:
            The calculated loss multiplied by its weight.
        """
        with self._state.scope(microbatch_idx) as state:
            computation = self._task.compute_loss(
                ComputeLossContext(pipeline_results=pipeline_outputs, state=state, schedule=self._schedule)
            )

            loss = computation.loss
            loss_weight = computation.loss_weight
            if loss_weight is None:
                loss_weight = torch.ones_like(loss)

            self._gradient_manager.add_loss_with_weight(loss.detach(), loss_weight.detach())
            self._task.update_metrics(UpdateMetricsContext(state=state, metrics=self._metrics.children))

        return loss * loss_weight


class TrainTaskOperator:
    """Orchestrates the forward and backward passes for a training task over one pack.

    It builds the per-microbatch inputs, reconfigures the pipeline schedule for the pack length, and
    drives execution. Loss/weight and metrics accumulate per microbatch through the loss callback.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        task: TrainTask,
        pipeline: PipelineScheduleInfo,
        pipeline_state: PipelineStateHandler,
        gradient_manager: GradientManager,
        job_schedule: JobSchedule,
        metrics: ComposeMetric,
    ):
        """Constructs the TrainTaskOperator.

        Args:
            dist_context: The distributed context.
            task: The user-defined training task logic.
            pipeline: Information about the pipeline schedule.
            pipeline_state: Handler for transient per-microbatch state during the step.
            gradient_manager: Gradient accumulator; told how many backward passes this step performs.
            job_schedule: Component tracking current step and progress.
            metrics: Metric collection updated per microbatch.
        """
        self._dist_context = dist_context
        self._task = task
        self._pipeline = pipeline
        self._pipeline_state = pipeline_state
        self._gradient_manager = gradient_manager
        self._job_schedule = job_schedule
        self._metrics = metrics

    def forward_backward(self, pack: MicrobatchPack) -> None:
        """Executes the forward and backward passes for one pack of microbatches.

        Args:
            pack: The step's pack of raw microbatches.
        """
        try:
            inputs_microbatches, kwargs_microbatches = build_pipeline_microbatch_inputs(
                self._task, self._pipeline_state, pack
            )
            self._gradient_manager.set_required_accumulations(len(pack))
            self._pipeline.schedule.step(
                inputs_microbatches=inputs_microbatches,
                kwargs_microbatches=kwargs_microbatches,
                callback=LossComputer(
                    state=self._pipeline_state,
                    task=self._task,
                    schedule=self._job_schedule,
                    gradient_manager=self._gradient_manager,
                    metrics=self._metrics,
                ),
            )
        finally:
            self._pipeline_state.reset()
