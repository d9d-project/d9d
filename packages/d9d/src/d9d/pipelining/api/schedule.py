import abc
import typing

from .types import PipelineStepHandler, TPipelineInput, TPipelineOutput, TSharedInput


class PipelineSchedule(abc.ABC, typing.Generic[TPipelineInput, TSharedInput, TPipelineOutput]):
    """Abstract base class for pipeline execution schedules.

    Type parameters:
        TPipelineInput: The ``PipelineInput`` fed to the first stage.
        TSharedInput: The ``SharedInput`` fed to every stage.
        TPipelineOutput: The ``PipelineOutput`` produced by the last stage and handed to the step handler.
    """

    @abc.abstractmethod
    def step(
        self,
        inputs_microbatches: tuple[TPipelineInput, ...],
        shared_microbatches: tuple[TSharedInput, ...],
        handler: PipelineStepHandler[TPipelineOutput],
    ):
        """Executes a single pipeline step over one pack of microbatches.

        The number of microbatches in the step is ``len(inputs_microbatches)`` and may vary between steps.

        Args:
            inputs_microbatches: Per-microbatch ``PipelineInput`` (fed to the first pipeline stage).
            shared_microbatches: Per-microbatch ``SharedInput`` (fed to every pipeline stage).
            handler: The code that the step runs around each model compute and on the outputs of the last stage.
        """
        ...
