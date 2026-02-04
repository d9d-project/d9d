import abc
from typing import Generic, TypeVar

import torch

from d9d.internals.pipeline_state import PipelineStateHandler
from d9d.loop.control import ComputeLossContext, InferenceTask, ProcessOutputsContext, TrainTask

from .stepper import Stepper

STATE_LOSS = "__internal_loss"
STATE_LOSS_WEIGHT = "__internal_loss_weight"


TOutput = TypeVar("TOutput")


class PipelineOutputsProcessor(abc.ABC, Generic[TOutput]):
    @abc.abstractmethod
    def __call__(
            self,
            pipeline_outputs: dict[str, torch.Tensor],
            microbatch_idx: int
    ) -> TOutput:
        ...


class LossComputer(PipelineOutputsProcessor[torch.Tensor]):
    """
    Handles the computation of loss values and their integration into the pipeline state.

    This component acts as a bridge between the raw outputs of the model pipeline
    and the user-defined training task. It retrieves the appropriate state context
    (potentially sharded per microbatch), executes the user's loss logic, persists
    metrics into the state for logging, and returns the loss*weight term for backpropagation.
    """

    def __init__(
            self,
            state: PipelineStateHandler,
            task: TrainTask,
            stepper: Stepper
    ):
        """
        Constructs a new LossComputer.

        Args:
            state: Handler for managing global and sharded pipeline states.
            task: The user-defined training task containing loss computation logic.
            stepper: Component tracking current step and progress.
        """

        self._state = state
        self._task = task
        self._stepper = stepper

    def __call__(
            self,
            pipeline_outputs: dict[str, torch.Tensor],
            microbatch_idx: int
    ) -> torch.Tensor:
        """
        Computes the weighted loss for a specific sharded microbatch or the full microbatch.

        This method retrieves the appropriate state context based on the microbatch
        index, delegates calculation to the training task, saves the raw loss and
        weight into the state for later retrieval, and returns the final scalar
        product used for backward passes.

        You can retrieve states by using `STATE_LOSS` and `STATE_LOSS_WEIGHT` keys.

        Args:
            pipeline_outputs: Dictionary containing model output tensors.
            microbatch_idx: Index of the current microbatch, or `None` for full microbatch execution.

        Returns:
            The calculated loss multiplied by its weight.
        """

        state = self._state.sharded_state(
            shard_id=microbatch_idx
        )

        computation = self._task.compute_loss(ComputeLossContext(
            pipeline_results=pipeline_outputs,
            state=state,
            stepper=self._stepper
        ))

        loss = computation.loss
        loss_weight = computation.loss_weight

        if loss_weight is None:
            loss_weight = torch.ones_like(loss)

        state[STATE_LOSS] = loss[None]
        state[STATE_LOSS_WEIGHT] = loss_weight[None]

        return loss * loss_weight


class InferenceProcessor(PipelineOutputsProcessor[None]):
    """
    Handles the processing of model outputs during inference or evaluation.

    This component retrieves the appropriate state context
    and delegates the output processing logic to the user-defined inference task.
    """

    def __init__(
            self,
            state: PipelineStateHandler,
            task: InferenceTask
    ):
        """
        Constructs a new ModelOutputsProcessor.

        Args:
            state: Handler for managing global and sharded pipeline states.
            task: The user-defined inference task containing processing logic.
        """

        self._state = state
        self._task = task

    def __call__(
            self,
            pipeline_outputs: dict[str, torch.Tensor],
            microbatch_idx: int
    ) -> None:
        """
        Processes model outputs for a specific microbatch or full batch.

        This method retrieves the relevant state (scoped by microbatch index if provided)
        and invokes the task's output processing logic.

        Args:
            pipeline_outputs: Dictionary containing model output tensors.
            microbatch_idx: Index of the current microbatch, or None if not using microbatching.
        """

        state = self._state.sharded_state(
            shard_id=microbatch_idx
        )

        self._task.process_outputs(ProcessOutputsContext(
            pipeline_results=pipeline_outputs,
            state=state
        ))
