from __future__ import annotations

import abc
import dataclasses
from typing import Generic, TypeVar

from torch import nn

from d9d.core.dist_context import DistributedContext
from d9d.core.types import ScalarTree
from d9d.loop.event import EventBus
from d9d.model_state.mapper import ModelStateMapper
from d9d.pipelining.api import PipelineStageInfo


@dataclasses.dataclass(kw_only=True)
class InitializeModelStageContext:
    """Context data required for initializing a specific model pipeline stage.

    Attributes:
        dist_context: The distributed execution context.
        stage: Metadata describing the current pipeline stage being initialized.
    """

    dist_context: DistributedContext
    stage: PipelineStageInfo


TModel = TypeVar("TModel", bound=nn.Module)


@dataclasses.dataclass(kw_only=True)
class InitializeModelStageResult(Generic[TModel]):
    """The result of initializing a model stage.

    Attributes:
        model: The PyTorch module.
        state_mapper: The mapper defining how to load weights into this module.
    """

    model: TModel
    state_mapper: ModelStateMapper


@dataclasses.dataclass(kw_only=True)
class ParallelizeModelStageContext(Generic[TModel]):
    """Context data required for horizontally parallelizing a model stage.

    Attributes:
        dist_context: The distributed execution context.
        stage: Metadata describing the current pipeline stage.
        model: The PyTorch module to be parallelized.
    """

    dist_context: DistributedContext
    stage: PipelineStageInfo
    model: TModel


@dataclasses.dataclass(kw_only=True)
class PrepareExportModelStageContext(Generic[TModel]):
    """Context data required for preparing a model stage for export.

    Attributes:
        dist_context: The distributed execution context.
        model: The PyTorch module to be exported.
    """

    dist_context: DistributedContext
    model: TModel


@dataclasses.dataclass(kw_only=True)
class PrepareExportModelStageResult:
    """The result of preparing a model stage for export.

    Attributes:
        state_mapper: The mapper defining how model parameters map to disk storage.
    """

    state_mapper: ModelStateMapper


@dataclasses.dataclass(kw_only=True)
class RegisterModelEventsContext:
    """Context for registering model-specific events.

    Attributes:
        dist_context: The distributed execution context.
        event_bus: The event bus for subscribing to events.
    """

    dist_context: DistributedContext
    event_bus: EventBus


class ModelProvider(abc.ABC, Generic[TModel]):
    """Abstract interface for defining the lifecycle of a distributed model.

    The provider initializes, parallelizes (shards, replicates, etc.) and prepares the model for export.
    """

    @abc.abstractmethod
    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult[TModel]:
        """Initializes the model architecture for a specific pipeline stage.

        It constructs the ``nn.Module`` for the requested stage.

        Construction runs on the meta device, so the method must not load weights. Instead, it returns a
        ``ModelStateMapper`` that maps checkpoint weights to the parameters of the new module.

        The method can change the architecture, e.g. inject LoRA adapters, if the returned mapper
        reflects the new structure.

        Args:
            context: Context for this operation.

        Returns:
            The model stage and its state mapper.
        """
        ...

    @abc.abstractmethod
    def parallelize_model_stage(self, context: ParallelizeModelStageContext[TModel]):
        """Converts the model parameters into distributed tensors (``DTensor``).

        Implementations must modify the model in place. They replicate or shard each parameter
        according to the chosen parallelism strategies.

        Args:
            context: Context for this operation.
        """

    @abc.abstractmethod
    def prepare_export_model_stage(
        self, context: PrepareExportModelStageContext[TModel]
    ) -> PrepareExportModelStageResult:
        """Prepares the state mapper required for saving the model to disk.

        The mapper defines how the in-memory model structure maps back to the checkpoint format.

        Args:
            context: Context for this operation.

        Returns:
            The state mapper for export.
        """

    def register_events(self, context: RegisterModelEventsContext) -> None:
        """Registers model-specific event subscriptions.

        Args:
            context: Context with the distributed context and the event bus.
        """

    def dump_hparams(self) -> ScalarTree:
        """Exports hyperparameters associated with this model for logging.

        Returns:
            A dictionary of hyperparameter names and values.
        """
        return {}
