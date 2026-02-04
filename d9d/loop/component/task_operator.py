import dataclasses

import torch

from d9d.core.dist_context import DistributedContext
from d9d.core.types import PyTree
from d9d.internals.pipeline_state import PipelineStateHandler
from d9d.loop.control import (
    BaseTask,
    BuildForwardInputsContext,
    InferenceTask,
    TrainTask,
    UpdateMetricsContext,
)
from d9d.metric.impl import ComposeMetric
from d9d.pipelining.factory.factory import PipelineScheduleInfo

from .pipeline_result_processing import STATE_LOSS, STATE_LOSS_WEIGHT


@dataclasses.dataclass(kw_only=True)
class ForwardResult:
    """
    Encapsulates the scalar results of a forward pass step.

    Attributes:
        loss: The computed loss tensor.
        loss_weight: The weight associated with this loss (usually batch size).
    """

    loss: torch.Tensor
    loss_weight: torch.Tensor


def _run_pipeline(
        task: BaseTask,
        pipeline: PipelineScheduleInfo,
        pipeline_state: PipelineStateHandler,
        batch: PyTree
):
    model_inputs = task.build_forward_inputs(
        BuildForwardInputsContext(
            batch=batch,
            state=pipeline_state.global_state()
        )
    )
    pipeline.schedule.configure_buffers(
        inputs=model_inputs.inputs,
        kwargs=model_inputs.kwargs,
        sharding_spec=model_inputs.pipeline_sharding_spec
    )
    pipeline.schedule.step(
        inputs=model_inputs.inputs,
        kwargs=model_inputs.kwargs
    )


class TrainTaskOperator:
    """
    Orchestrates the execution of the forward and backward passes for a specific training task.

    It manages input construction, schedule execution,
    loss computation, and metric updates within the lifecycle of a single step.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            task: TrainTask,
            pipeline: PipelineScheduleInfo,
            pipeline_state: PipelineStateHandler,
            metrics: ComposeMetric
    ):
        """
        Constructs the TrainTaskOperator.

        Args:
            dist_context: The distributed context.
            task: The user-defined training task logic.
            pipeline: Information about the pipeline schedule.
            pipeline_state: Handler for transient state storage during the step.
            metrics: Metric collection to update after the pass.
        """

        self._dist_context = dist_context
        self._task = task
        self._pipeline = pipeline
        self._pipeline_state = pipeline_state
        self._metrics = metrics

    def forward_backward(self, batch: PyTree) -> ForwardResult | None:
        """
        Executes the forward and backward passes for a single batch.

        This method handles:

        1. Context preparation and input building via the `TrainTask`.
        2. Execution via Pipeline Parallel schedule or standard Forward/Backward.
        3. Metric updates based on the results.
        4. Reliable cleanup of the pipeline state.

        Args:
            batch: The input batch data.

        Returns:
            A `ForwardResult` containing the loss and weight if this rank is responsible
            for loss calculation (e.g., the last pipeline stage or in standard DP).
            Returns `None` if this rank is an intermediate pipeline stage that does
            not compute loss.
        """

        try:
            # Do forward and backward pass
            _run_pipeline(
                pipeline_state=self._pipeline_state,
                task=self._task,
                pipeline=self._pipeline,
                batch=batch
            )

            # Update metrics if possible

            pipeline_state = self._pipeline_state.global_state()
            if not self._pipeline.has_last_stage:
                return None

            self._task.update_metrics(UpdateMetricsContext(
                state=pipeline_state,
                metrics=self._metrics.children
            ))
            return ForwardResult(
                loss=pipeline_state[STATE_LOSS],
                loss_weight=pipeline_state[STATE_LOSS_WEIGHT]
            )
        finally:
            self._pipeline_state.reset()


class InferenceTaskOperator:
    """
    Orchestrates the execution of the forward pass for a specific inference task.

    It manages input
    construction, schedule execution, and state lifecycle management.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            task: InferenceTask,
            pipeline: PipelineScheduleInfo,
            pipeline_state: PipelineStateHandler
    ):
        """
        Constructs the InferenceTaskOperator.

        Args:
            dist_context: The distributed context.
            task: The user-defined inference task logic.
            pipeline: Information about the pipeline schedule.
            pipeline_state: Handler for transient state storage during the step.
        """

        self._dist_context = dist_context
        self._task = task
        self._pipeline = pipeline
        self._pipeline_state = pipeline_state

    def forward(self, batch: PyTree) -> None:
        """
        Executes the forward pass for a single batch.

        This method handles:

        1. Context preparation and input building via the `InferenceTask`.
        2. Execution via Pipeline Parallel schedule.
        3. Reliable cleanup of the pipeline state.

        Args:
            batch: The input batch data.
        """

        try:
            # Do forward pass
            _run_pipeline(
                pipeline_state=self._pipeline_state,
                task=self._task,
                pipeline=self._pipeline,
                batch=batch
            )
        finally:
            self._pipeline_state.reset()
