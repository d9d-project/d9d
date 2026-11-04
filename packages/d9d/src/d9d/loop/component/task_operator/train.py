import typing
from contextlib import AbstractContextManager

import torch
from torch.profiler import record_function

from d9d.data import MicrobatchPack
from d9d.dist_context import BATCH_DOMAIN, DistributedContext
from d9d.metric.catalogue.container import ComposeMetric
from d9d.pipelining.api import PipelineComputeKind, PipelineStepHandler
from d9d.pipelining.factory import PipelineScheduleInfo
from d9d.pytree import PyTree
from d9d.rng import fork_rng_child

from ...control import ComputeLossContext, TrainTask, UpdateMetricsContext
from ..gradient_manager import GradientManager
from ..job_schedule import JobSchedule
from ..pipeline_state import PipelineStateHandler
from .common import build_pipeline_microbatch_inputs

TBatch = typing.TypeVar("TBatch", bound=PyTree)
TPipelineInput = typing.TypeVar("TPipelineInput")
TSharedInput = typing.TypeVar("TSharedInput")
TPipelineOutput = typing.TypeVar("TPipelineOutput")
TState = typing.TypeVar("TState", bound=PyTree)


class TrainStepHandler(PipelineStepHandler[TPipelineOutput], typing.Generic[TPipelineOutput, TState]):
    """The step handler that computes and accumulates the training loss for each microbatch of a step.

    It bridges the raw outputs of the model pipeline and the user-defined training task.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        state: PipelineStateHandler[TState],
        task: TrainTask[typing.Any, typing.Any, typing.Any, TPipelineOutput, TState],
        schedule: JobSchedule,
        gradient_manager: GradientManager,
        metrics: ComposeMetric,
    ):
        """Constructs the ``TrainStepHandler`` object.

        Args:
            dist_context: The distributed context.
            state: Handler for the per-microbatch pipeline state.
            task: The user-defined training task containing loss computation logic.
            schedule: The job schedule that tracks the current step.
            gradient_manager: Accumulator of loss/weight for gradient reduction.
            metrics: Metric collection updated per microbatch.
        """
        self._cp_rank = (
            dist_context.mesh_for(BATCH_DOMAIN)["cp"].get_local_rank() if dist_context.mesh_params.is_distributed else 0
        )
        self._state = state
        self._task = task
        self._schedule = schedule
        self._gradient_manager = gradient_manager
        self._metrics = metrics

    def compute_context(
        self, kind: PipelineComputeKind, stage_idx: int, microbatch_idx: int
    ) -> AbstractContextManager[None]:
        # The model compute of the cp ranks works on other parts of the same samples, so it needs other numbers.
        return fork_rng_child("cp", self._cp_rank, kind.value, stage_idx, microbatch_idx)

    def process_outputs(self, outputs: TPipelineOutput, microbatch_idx: int) -> torch.Tensor:
        """Computes the weighted loss for a microbatch and accumulates its loss/weight and metrics.

        Args:
            outputs: The ``PipelineOutput`` produced by the last stage.
            microbatch_idx: Index of the current microbatch within the step's pack.

        Returns:
            The calculated loss multiplied by its weight.
        """
        with self._state.scope(microbatch_idx) as state:
            with record_function("Compute Loss"), fork_rng_child("compute_loss", microbatch_idx):
                computation = self._task.compute_loss(
                    ComputeLossContext(pipeline_results=outputs, state=state, schedule=self._schedule)
                )

            loss = computation.loss
            loss_weight = computation.loss_weight
            if loss_weight is None:
                loss_weight = torch.ones_like(loss)

            self._gradient_manager.add_loss_with_weight(loss.detach(), loss_weight.detach())
            with record_function("Update Metrics"), fork_rng_child("update_metrics", microbatch_idx):
                self._task.update_metrics(UpdateMetricsContext(state=state, metrics=self._metrics.children))

        return loss * loss_weight


class TrainTaskOperator(typing.Generic[TBatch, TPipelineInput, TSharedInput, TPipelineOutput, TState]):
    """Operator that orchestrates the forward and backward passes for a training task over one pack.

    It builds the per-microbatch inputs, reconfigures the pipeline schedule for the pack length, and
    drives execution. Loss/weight and metrics accumulate per microbatch through the step handler.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        task: TrainTask[TBatch, TPipelineInput, TSharedInput, TPipelineOutput, TState],
        pipeline: PipelineScheduleInfo[TPipelineInput, TSharedInput, TPipelineOutput],
        pipeline_state: PipelineStateHandler[TState],
        gradient_manager: GradientManager,
        job_schedule: JobSchedule,
        metrics: ComposeMetric,
    ):
        """Constructs the ``TrainTaskOperator`` object.

        Args:
            dist_context: The distributed context.
            task: The user-defined training task logic.
            pipeline: Information about the pipeline schedule.
            pipeline_state: Handler for transient per-microbatch state during the step.
            gradient_manager: Gradient accumulator; told how many backward passes this step performs.
            job_schedule: The job schedule that tracks the current step.
            metrics: Metric collection updated per microbatch.
        """
        self._dp_rank = (
            dist_context.mesh_for(BATCH_DOMAIN)["dp"].get_local_rank() if dist_context.mesh_params.is_distributed else 0
        )
        self._task = task
        self._pipeline = pipeline
        self._pipeline_state = pipeline_state
        self._gradient_manager = gradient_manager
        self._step_handler = TrainStepHandler(
            dist_context=dist_context,
            state=pipeline_state,
            task=task,
            schedule=job_schedule,
            gradient_manager=gradient_manager,
            metrics=metrics,
        )

    def forward_backward(self, pack: MicrobatchPack[TBatch]) -> None:
        """Executes the forward and backward passes for one pack of microbatches.

        Args:
            pack: The step's pack of raw microbatches.
        """
        # The dp ranks work on other samples, so the task hooks and the model need other numbers.
        with record_function("Forward & Backward"), fork_rng_child("dp", self._dp_rank):
            try:
                inputs_microbatches, shared_microbatches = build_pipeline_microbatch_inputs(
                    task=self._task, pipeline_state=self._pipeline_state, pack=pack
                )
                self._gradient_manager.set_required_accumulations(len(pack))
                self._pipeline.schedule.step(
                    inputs_microbatches=inputs_microbatches,
                    shared_microbatches=shared_microbatches,
                    handler=self._step_handler,
                )
            finally:
                self._pipeline_state.reset()
