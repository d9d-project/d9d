import torch

from d9d.core.dist_context import DistributedContext
from d9d.core.types import MicrobatchPack
from d9d.loop.control import InferenceTask, ProcessOutputsContext
from d9d.pipelining.factory.factory import PipelineScheduleInfo

from ..pipeline_state import PipelineStateHandler
from .common import build_pipeline_microbatch_inputs


class InferenceProcessor:
    """Handles the processing of model outputs during inference or evaluation.

    This component retrieves the per-microbatch state and delegates the output processing logic to the
    user-defined inference task.
    """

    def __init__(self, state: PipelineStateHandler, task: InferenceTask):
        """Constructs a new InferenceProcessor.

        Args:
            state: Handler for the per-microbatch pipeline state.
            task: The user-defined inference task containing processing logic.
        """
        self._state = state
        self._task = task

    def __call__(self, pipeline_outputs: dict[str, torch.Tensor], microbatch_idx: int) -> None:
        """Processes model outputs for a specific microbatch.

        Args:
            pipeline_outputs: Dictionary containing model output tensors.
            microbatch_idx: Index of the current microbatch within the step's pack.
        """
        with self._state.scope(microbatch_idx) as state:
            self._task.process_outputs(ProcessOutputsContext(pipeline_results=pipeline_outputs, state=state))


class InferenceTaskOperator:
    """Orchestrates the forward pass for an inference task over one pack."""

    def __init__(
        self,
        dist_context: DistributedContext,
        task: InferenceTask,
        pipeline: PipelineScheduleInfo,
        pipeline_state: PipelineStateHandler,
    ):
        """Constructs the InferenceTaskOperator.

        Args:
            dist_context: The distributed context.
            task: The user-defined inference task logic.
            pipeline: Information about the pipeline schedule.
            pipeline_state: Handler for transient per-microbatch state during the step.
        """
        self._dist_context = dist_context
        self._task = task
        self._pipeline = pipeline
        self._pipeline_state = pipeline_state

    def forward(self, pack: MicrobatchPack) -> None:
        """Executes the forward pass for one pack of microbatches.

        Args:
            pack: The step's pack of raw microbatches.
        """
        try:
            inputs_microbatches, kwargs_microbatches = build_pipeline_microbatch_inputs(
                self._task, self._pipeline_state, pack
            )
            self._pipeline.schedule.step(
                inputs_microbatches=inputs_microbatches,
                kwargs_microbatches=kwargs_microbatches,
                callback=InferenceProcessor(task=self._task, state=self._pipeline_state),
            )
        finally:
            self._pipeline_state.reset()
