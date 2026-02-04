import dataclasses

import torch

from d9d.core.dist_context import DistributedContext
from d9d.core.types import PyTree
from d9d.internals.pipeline_state import PipelineStateHandler
from d9d.loop.control import BuildForwardInputsContext, BuildForwardInputsResult, TrainTask, UpdateMetricsContext
from d9d.metric.impl import ComposeMetric
from d9d.pipelining.factory.factory import PipelineScheduleInfo

from .loss_computer import STATE_LOSS, STATE_LOSS_WEIGHT, LossComputer
from .model_stage_factory import TrackedModules


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


class TrainTaskOperator:
    """
    Orchestrates the execution of the forward and backward passes for a specific training task.

    This class abstracts the difference between standard execution
    and pipeline-parallel execution. It manages input construction, schedule execution,
    loss computation, and metric updates within the lifecycle of a single step.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            task: TrainTask,
            pp_schedule: PipelineScheduleInfo | None,
            tracked_modules: TrackedModules,
            loss_computer: LossComputer,
            pipeline_state: PipelineStateHandler,
            metrics: ComposeMetric
    ):
        """
        Constructs the TrainTaskOperator.

        Args:
            dist_context: The distributed context.
            task: The user-defined training task logic.
            pp_schedule: Information about the pipeline schedule.
            tracked_modules: The model modules being trained.
            loss_computer: Component responsible for calculating loss from outputs.
            pipeline_state: Handler for transient state storage during the step.
            metrics: Metric collection to update after the pass.
        """

        self._dist_context = dist_context
        self._task = task
        self._pp_schedule = pp_schedule
        self._tracked_modules = tracked_modules
        self._loss_computer = loss_computer
        self._pipeline_state = pipeline_state
        self._metrics = metrics

    def _forward_backward_pipelining(self, model_inputs: BuildForwardInputsResult):
        if self._pp_schedule is None:
            raise ValueError("Cannot run pipelined pass if pipelining is disabled")

        self._pp_schedule.schedule.configure_buffers(
            inputs=model_inputs.inputs,
            kwargs=model_inputs.kwargs,
            sharding_spec=model_inputs.pipeline_sharding_spec
        )
        self._pp_schedule.schedule.step(
            inputs=model_inputs.inputs,
            kwargs=model_inputs.kwargs
        )

    def _forward_backward_regular(self, model_inputs: BuildForwardInputsResult):
        pipeline_outputs = self._tracked_modules(
            **model_inputs.inputs,
            **model_inputs.kwargs
        )
        loss = self._loss_computer.compute_loss_mul_weight(
            pipeline_outputs=pipeline_outputs,
            microbatch_idx=None
        )
        # free to avoid bwd peaking memory
        del pipeline_outputs
        loss.backward()

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
            model_inputs = self._task.build_forward_inputs(
                BuildForwardInputsContext(
                    batch=batch,
                    state=self._pipeline_state.global_state()
                )
            )

            if self._dist_context.mesh_params.has_pipeline_parallel:
                self._forward_backward_pipelining(model_inputs)
            else:
                self._forward_backward_regular(model_inputs)

            # Update metrics if possible

            pipeline_state = self._pipeline_state.global_state()

            if (
                    self._dist_context.mesh_params.has_pipeline_parallel and
                    self._pp_schedule is not None and
                    not self._pp_schedule.has_last_stage
            ):
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
