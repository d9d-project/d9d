import abc
import dataclasses
import typing
from collections.abc import Mapping
from typing import Any, Protocol

import torch
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext
from d9d.core.types import PyTree, ScalarTree
from d9d.pipelining.api import PipelineShardingSpec

if typing.TYPE_CHECKING:
    from d9d.internals.pipeline_state import PipelineState
    from d9d.loop.component import Stepper
    from d9d.metric import Metric


TBatch = typing.TypeVar("TBatch", bound=PyTree)


@dataclasses.dataclass(kw_only=True)
class BuildForwardInputsContext(typing.Generic[TBatch]):
    """
    Context data to prepare inputs for the model forward pass.

    Attributes:
        batch: The raw batch data loaded from the DataLoader object.
        state: The current state of the pipeline. You can assign any data to this state object, and it will be
            accessible during this pipeline step (e.g. when computing loss)
    """

    batch: TBatch
    state: "PipelineState"


@dataclasses.dataclass(kw_only=True)
class BuildForwardInputsResult:
    """
    The result of processing the raw batch into model inputs.

    Attributes:
        inputs: A dictionary of inputs that are passed to model pipeline as input data
            (first stage only if using pipeline parallelism).
        kwargs: A dictionary of keyword arguments passed to each pipeline stage.
        pipeline_sharding_spec: A specification defining how inputs and kwargs should be split
            into micro-batches for pipeline parallelism. If None, the framework assumes
            standard behavior where all the non-scalar Tensors and lists are split by 0 dimension.
    """

    inputs: dict[str, torch.Tensor]
    kwargs: dict[str, Any]
    pipeline_sharding_spec: PipelineShardingSpec | None = None


@dataclasses.dataclass(kw_only=True)
class FinalizeContext:
    """Context data provided when the task is being finalized."""


class BaseTask(abc.ABC, Stateful, typing.Generic[TBatch]):
    """Abstract base class representing a unit of work (Task) in the training/inference loop."""

    @abc.abstractmethod
    def build_forward_inputs(self, ctx: BuildForwardInputsContext[TBatch]) -> BuildForwardInputsResult:
        """
        Transforms raw data loaded from the DataLoader into arguments for the model.

        Args:
            ctx: Context object.

        Returns:
            Result object.
        """

        ...

    def state_dict(self) -> dict[str, Any]:
        """
        Returns the state dictionary for checkpointing this task.

        Returns:
            A dictionary containing the task's state.
        """

        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restores the task's state from the provided dictionary.

        Args:
            state_dict: The state dictionary to load.
        """
        # do nothing by default

    def finalize(self, ctx: FinalizeContext) -> None:
        """
        Performs cleanup or final actions when the task execution finishes.

        Args:
             ctx: Context object.
        """


@dataclasses.dataclass(kw_only=True)
class ComputeLossContext:
    """
    Context data provided to calculate the loss during training.

    Attributes:
        pipeline_results: The outputs returned by the model's forward pass.
        state: The current state of the pipeline. You can assign any data to this state object, and it will be
            accessible during this pipeline step (e.g. when calculating metrics)
        stepper: Component tracking the current step.
    """

    pipeline_results: Mapping[str, torch.Tensor]
    state: "PipelineState"
    stepper: "Stepper"


@dataclasses.dataclass(kw_only=True)
class ComputeLossResult:
    """
    The result of the loss computation.

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
    """
    Result of metric initialization.

    Attributes:
        metrics: A dictionary mapping metric names to Metric instances.
    """

    metrics: dict[str, "Metric"]


@dataclasses.dataclass(kw_only=True)
class UpdateMetricsContext:
    """
    Context data provided to update metrics after a step.

    Attributes:
        state: The current state of the pipeline.
        metrics: The dictionary of metrics to be updated.
    """

    state: "PipelineState"
    metrics: Mapping[str, "Metric"]


class TrainTask(BaseTask, abc.ABC, typing.Generic[TBatch]):
    """Abstract base class for defining training-specific logic."""

    @abc.abstractmethod
    def compute_loss(self, ctx: ComputeLossContext) -> ComputeLossResult:
        """
        Calculates the loss based on model outputs.

        Args:
            ctx: Context object.

        Returns:
            Result object.
        """

        ...

    def create_metrics(self, ctx: CreateMetricsContext) -> CreateMetricsResult:
        """
        Initializes metrics to be tracked during training.

        Args:
             ctx: Context object.

        Returns:
            Result object.
        """

        return CreateMetricsResult(metrics={})

    def update_metrics(self, ctx: UpdateMetricsContext):
        """
        Updates the state of the metrics at the end of training step.

        Args:
            ctx: Context object.
        """

    def dump_hparams(self) -> ScalarTree:
        """
        Exports hyperparameters associated with this task for logging.

        Returns:
            A dictionary of hyperparameter names and values.
        """

        return {}


@dataclasses.dataclass(kw_only=True)
class TrainTaskProviderContext:
    """
    Context data provided to the factory creating a TrainTask.

    Attributes:
        dist_context: Information about the distributed environment.
    """

    dist_context: DistributedContext


@typing.runtime_checkable
class TrainTaskProvider(Protocol):
    """Protocol that creates a TrainTask instance."""

    def __call__(self, ctx: TrainTaskProviderContext) -> TrainTask:
        """
        Creates and returns a new TrainTask.

        Args:
            ctx: Context object.

        Returns:
            An instantiated TrainTask.
        """

        ...


@dataclasses.dataclass(kw_only=True)
class ProcessOutputsContext:
    """
    Context data provided to process outputs during inference.

    Attributes:
        outputs: The outputs returned by the model's forward pass.
        state: The current state of the pipeline.
    """

    outputs: dict[str, torch.Tensor]
    state: "PipelineState"


class InferenceTask(BaseTask, abc.ABC, typing.Generic[TBatch]):
    """Abstract base class for defining inference-specific logic."""

    @abc.abstractmethod
    def process_outputs(self, ctx: ProcessOutputsContext):
        """
        Processes the model outputs (e.g. saving to disk, decoding tokens).

        Args:
            ctx: Context containing the model outputs and pipeline state.
        """

        ...


@dataclasses.dataclass(kw_only=True)
class InferenceTaskProviderContext:
    """
    Context data provided to the factory creating an InferenceTask.

    Attributes:
        dist_context: Information about the distributed environment.
    """

    dist_context: DistributedContext


@typing.runtime_checkable
class InferenceTaskProvider(Protocol):
    """Protocol for a callable that creates an InferenceTask instance."""

    def __call__(self, ctx: InferenceTaskProviderContext) -> InferenceTask:
        """
        Creates and returns a new InferenceTask.

        Args:
            ctx: Context providing distributed environment information.

        Returns:
            An instantiated InferenceTask.
        """
        ...
