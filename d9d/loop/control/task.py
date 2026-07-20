import abc
import dataclasses
import typing
from collections.abc import Mapping
from typing import Any, Protocol

import torch
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext
from d9d.core.types import PyTree, ScalarTree
from d9d.loop.event import EventBus

if typing.TYPE_CHECKING:
    from d9d.loop.component import JobSchedule
    from d9d.metric import Metric


TBatch = typing.TypeVar("TBatch", bound=PyTree)
TState = typing.TypeVar("TState", bound=PyTree)
TPipelineInput = typing.TypeVar("TPipelineInput")
TSharedInput = typing.TypeVar("TSharedInput")
TPipelineOutput = typing.TypeVar("TPipelineOutput")


@dataclasses.dataclass(kw_only=True)
class BuildForwardInputsContext(typing.Generic[TBatch]):
    """Context data to prepare inputs for the model forward pass of a single microbatch.

    Attributes:
        batch: One raw microbatch of data produced by the data stream.
    """

    batch: TBatch


@dataclasses.dataclass(kw_only=True)
class BuildForwardInputsResult(typing.Generic[TPipelineInput, TSharedInput, TState]):
    """The result of processing one raw microbatch into model inputs.

    Attributes:
        input: The ``PipelineInput`` passed to the model pipeline as input data
            (first stage only if using pipeline parallelism).
        shared: The ``SharedInput`` passed to every pipeline stage.
        state: Side-data (a PyTree, e.g. a TypedDict) carried to loss/output processing for this same
            microbatch — labels, masks, anything the model forward does not return but the loss needs.
    """

    input: TPipelineInput
    shared: TSharedInput
    state: TState


@dataclasses.dataclass(kw_only=True)
class RegisterTaskEventsContext:
    """Context for registering task-specific events.

    Attributes:
        dist_context: The distributed execution context.
        event_bus: The event bus for subscribing to events.
    """

    dist_context: DistributedContext
    event_bus: EventBus


@dataclasses.dataclass(kw_only=True)
class FinalizeContext:
    """Context data provided when the task is being finalized."""


class BaseTask(abc.ABC, Stateful, typing.Generic[TBatch, TPipelineInput, TSharedInput, TState]):
    """Abstract base class representing a unit of work (Task) in the training/inference loop.

    Type parameters:
        TBatch: The raw microbatch type produced by the data stream.
        TPipelineInput: The ``PipelineInput`` fed to the first pipeline stage.
        TSharedInput: The ``SharedInput`` fed to every pipeline stage.
        TState: The per-microbatch side-data carried from input building to loss/output processing.
            Tasks that carry nothing use ``None``.
    """

    @abc.abstractmethod
    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext[TBatch]
    ) -> BuildForwardInputsResult[TPipelineInput, TSharedInput, TState]:
        """Transforms one raw microbatch into arguments for the model.

        Called once per microbatch in the step's pack.

        Args:
            ctx: Context object.

        Returns:
            Result object, including the ``state`` side-data carried to loss/output processing.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Returns the state dictionary for checkpointing this task.

        Returns:
            A dictionary containing the task's state.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the task's state from the provided dictionary.

        Args:
            state_dict: The state dictionary to load.
        """
        # do nothing by default

    def register_events(self, context: RegisterTaskEventsContext) -> None:
        """Register task-specific event subscriptions.

        Args:
            context: Context providing access to the distributed environment
                and the event bus.
        """

    def finalize(self, ctx: FinalizeContext) -> None:
        """Performs cleanup or final actions when the task execution finishes.

        Args:
             ctx: Context object.
        """


@dataclasses.dataclass(kw_only=True)
class ComputeLossContext(typing.Generic[TPipelineOutput, TState]):
    """Context data provided to calculate the loss during training.

    Attributes:
        pipeline_results: The ``PipelineOutput`` returned by the model's forward pass.
        state: The side-data this microbatch's ``build_forward_inputs`` returned.
        schedule: Component tracking the current step.
    """

    pipeline_results: TPipelineOutput
    state: TState
    schedule: "JobSchedule"


@dataclasses.dataclass(kw_only=True)
class ComputeLossResult:
    """The result of the loss computation.

    Attributes:
        loss: The scalar tensor representing the loss to be backpropagated.
        loss_weight: The weight to apply to the loss (for synchronizing gradients using weighted mean).
            None for 1.0.
    """

    loss: torch.Tensor
    loss_weight: torch.Tensor | None


@dataclasses.dataclass(kw_only=True)
class CreateMetricsContext:
    """Context data provided to initialize metrics."""


@dataclasses.dataclass(kw_only=True)
class CreateMetricsResult:
    """Result of metric initialization.

    Attributes:
        metrics: A dictionary mapping metric names to Metric instances.
    """

    metrics: dict[str, "Metric"]


@dataclasses.dataclass(kw_only=True)
class UpdateMetricsContext(typing.Generic[TState]):
    """Context data provided to update metrics after a step.

    Attributes:
        state: The side-data this microbatch's ``build_forward_inputs`` returned.
        metrics: The dictionary of metrics to be updated.
    """

    state: TState
    metrics: Mapping[str, "Metric"]


class TrainTask(
    BaseTask[TBatch, TPipelineInput, TSharedInput, TState],
    abc.ABC,
    typing.Generic[TBatch, TPipelineInput, TSharedInput, TPipelineOutput, TState],
):
    """Abstract base class for defining training-specific logic."""

    @abc.abstractmethod
    def compute_loss(self, ctx: ComputeLossContext[TPipelineOutput, TState]) -> ComputeLossResult:
        """Calculates the loss based on model outputs.

        Args:
            ctx: Context object.

        Returns:
            Result object.
        """
        ...

    def create_metrics(self, ctx: CreateMetricsContext) -> CreateMetricsResult:
        """Initializes metrics to be tracked during training.

        Args:
             ctx: Context object.

        Returns:
            Result object.
        """
        return CreateMetricsResult(metrics={})

    def update_metrics(self, ctx: UpdateMetricsContext[TState]):
        """Updates the state of the metrics at the end of training step.

        Args:
            ctx: Context object.
        """

    def dump_hparams(self) -> ScalarTree:
        """Exports hyperparameters associated with this task for logging.

        Returns:
            A dictionary of hyperparameter names and values.
        """
        return {}


@dataclasses.dataclass(kw_only=True)
class TrainTaskProviderContext:
    """Context data provided to the factory creating a TrainTask.

    Attributes:
        dist_context: Information about the distributed environment.
    """

    dist_context: DistributedContext


@typing.runtime_checkable
class TrainTaskProvider(Protocol):
    """Protocol that creates a TrainTask instance."""

    def __call__(self, ctx: TrainTaskProviderContext) -> TrainTask:
        """Creates and returns a new TrainTask.

        Args:
            ctx: Context object.

        Returns:
            An instantiated TrainTask.
        """
        ...


@dataclasses.dataclass(kw_only=True)
class ProcessOutputsContext(typing.Generic[TPipelineOutput, TState]):
    """Context data provided to process outputs during inference.

    Attributes:
        pipeline_results: The ``PipelineOutput`` returned by the model's forward pass.
        state: The side-data this microbatch's ``build_forward_inputs`` returned.
    """

    pipeline_results: TPipelineOutput
    state: TState


class InferenceTask(
    BaseTask[TBatch, TPipelineInput, TSharedInput, TState],
    abc.ABC,
    typing.Generic[TBatch, TPipelineInput, TSharedInput, TPipelineOutput, TState],
):
    """Abstract base class for defining inference-specific logic."""

    @abc.abstractmethod
    def process_outputs(self, ctx: ProcessOutputsContext[TPipelineOutput, TState]):
        """Processes the model outputs (e.g. saving to disk, decoding tokens).

        Args:
            ctx: Context containing the model outputs and pipeline state.
        """
        ...


@dataclasses.dataclass(kw_only=True)
class InferenceTaskProviderContext:
    """Context data provided to the factory creating an InferenceTask.

    Attributes:
        dist_context: Information about the distributed environment.
    """

    dist_context: DistributedContext


@typing.runtime_checkable
class InferenceTaskProvider(Protocol):
    """Protocol for a callable that creates an InferenceTask instance."""

    def __call__(self, ctx: InferenceTaskProviderContext) -> InferenceTask:
        """Creates and returns a new InferenceTask.

        Args:
            ctx: Context providing distributed environment information.

        Returns:
            An instantiated InferenceTask.
        """
        ...
