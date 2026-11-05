import typing
from contextlib import AbstractContextManager

from torch.profiler import record_function

from d9d.data import MicrobatchPack
from d9d.dist_context import BASE_DOMAIN, BaseDim, DistributedContext
from d9d.pipelining.api import PipelineComputeKind, PipelineStepHandler
from d9d.pipelining.factory import PipelineScheduleInfo
from d9d.pytree import PyTree
from d9d.rng import fork_rng_child

from ...control import InferenceTask, ProcessOutputsContext
from ..pipeline_state import PipelineStateHandler
from .common import build_pipeline_microbatch_inputs

TBatch = typing.TypeVar("TBatch", bound=PyTree)
TPipelineInput = typing.TypeVar("TPipelineInput")
TSharedInput = typing.TypeVar("TSharedInput")
TPipelineOutput = typing.TypeVar("TPipelineOutput")
TState = typing.TypeVar("TState", bound=PyTree)


class InferenceStepHandler(PipelineStepHandler[TPipelineOutput], typing.Generic[TPipelineOutput, TState]):
    """The step handler that processes model outputs during inference or evaluation.

    It retrieves the per-microbatch state and delegates the output processing logic to the
    user-defined inference task.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        state: PipelineStateHandler[TState],
        task: InferenceTask[typing.Any, typing.Any, typing.Any, TPipelineOutput, TState],
    ):
        """Constructs the ``InferenceStepHandler`` object.

        Args:
            dist_context: The distributed context.
            state: Handler for the per-microbatch pipeline state.
            task: The user-defined inference task containing processing logic.
        """
        self._cp_rank = dist_context.coordinate(BASE_DOMAIN, BaseDim.CP)
        self._state = state
        self._task = task

    def compute_context(
        self, kind: PipelineComputeKind, stage_idx: int, microbatch_idx: int
    ) -> AbstractContextManager[None]:
        # The model compute of the cp ranks works on other parts of the same samples, so it needs other numbers.
        return fork_rng_child("cp", self._cp_rank, kind.value, stage_idx, microbatch_idx)

    def process_outputs(self, outputs: TPipelineOutput, microbatch_idx: int) -> None:
        """Processes model outputs for a specific microbatch.

        Args:
            outputs: The ``PipelineOutput`` produced by the last stage.
            microbatch_idx: Index of the current microbatch within the step's pack.
        """
        with (
            record_function("Process Outputs"),
            fork_rng_child("process_outputs", microbatch_idx),
            self._state.scope(microbatch_idx) as state,
        ):
            self._task.process_outputs(ProcessOutputsContext(pipeline_results=outputs, state=state))


class InferenceTaskOperator(typing.Generic[TBatch, TPipelineInput, TSharedInput, TPipelineOutput, TState]):
    """Operator that orchestrates the forward pass for an inference task over one pack."""

    def __init__(
        self,
        dist_context: DistributedContext,
        task: InferenceTask[TBatch, TPipelineInput, TSharedInput, TPipelineOutput, TState],
        pipeline: PipelineScheduleInfo[TPipelineInput, TSharedInput, TPipelineOutput],
        pipeline_state: PipelineStateHandler[TState],
    ):
        """Constructs the ``InferenceTaskOperator`` object.

        Args:
            dist_context: The distributed context.
            task: The user-defined inference task logic.
            pipeline: Information about the pipeline schedule.
            pipeline_state: Handler for transient per-microbatch state during the step.
        """
        self._dp_rank = dist_context.coordinate(BASE_DOMAIN, BaseDim.DP)
        self._task = task
        self._pipeline = pipeline
        self._pipeline_state = pipeline_state
        self._step_handler = InferenceStepHandler(dist_context=dist_context, state=pipeline_state, task=task)

    def forward(self, pack: MicrobatchPack[TBatch]) -> None:
        """Executes the forward pass for one pack of microbatches.

        Args:
            pack: The step's pack of raw microbatches.
        """
        # The dp ranks work on other samples, so the task hooks and the model need other numbers.
        with record_function("Forward"), fork_rng_child("dp", self._dp_rank):
            try:
                inputs_microbatches, shared_microbatches = build_pipeline_microbatch_inputs(
                    task=self._task, pipeline_state=self._pipeline_state, pack=pack
                )
                self._pipeline.schedule.step(
                    inputs_microbatches=inputs_microbatches,
                    shared_microbatches=shared_microbatches,
                    handler=self._step_handler,
                )
            finally:
                self._pipeline_state.reset()
