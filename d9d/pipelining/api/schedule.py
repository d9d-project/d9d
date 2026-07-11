import abc
import typing

from .types import (
    PipelineLossFn,
    PipelineResultFn,
    TPipelineInput,
    TPipelineOutput,
    TSharedInput,
)


class PipelineSchedule(abc.ABC, typing.Generic[TPipelineInput, TSharedInput, TPipelineOutput]):
    """Abstract base class defining the interface for pipeline execution schedules.

    Type parameters:
        TPipelineInput: The ``PipelineInput`` fed to the first stage.
        TSharedInput: The ``SharedInput`` fed to every stage.
        TPipelineOutput: The ``PipelineOutput`` produced by the last stage and handed to the output processing callback.
    """

    @abc.abstractmethod
    def step(
        self,
        inputs_microbatches: tuple[TPipelineInput, ...],
        shared_microbatches: tuple[TSharedInput, ...],
        callback: PipelineLossFn[TPipelineOutput] | PipelineResultFn[TPipelineOutput],
    ):
        """Executes a single pipeline step over one pack of microbatches.

        The schedule receives the microbatches.
        The number of microbatches in the step is ``len(inputs_microbatches)`` and may vary between steps.

        Args:
            inputs_microbatches: Per-microbatch ``PipelineInput`` (fed to the first pipeline stage).
            shared_microbatches: Per-microbatch ``SharedInput`` (fed to every pipeline stage).
            callback: Function to compute loss or process pipeline results.
        """
        ...
