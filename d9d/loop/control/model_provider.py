import abc
import dataclasses
from typing import Generic, TypeVar

from torch import nn

from d9d.core.dist_context import DistributedContext
from d9d.core.types import ScalarTree
from d9d.model_state.mapper import ModelStateMapper
from d9d.pipelining.api import PipelineStageInfo


@dataclasses.dataclass(kw_only=True)
class InitializeModelStageContext:
    """
    Context data required for initializing a specific model pipeline stage.

    Attributes:
        dist_context: The distributed execution context.
        stage: Metadata describing the current pipeline stage being initialized.
    """

    dist_context: DistributedContext
    stage: PipelineStageInfo


TModel = TypeVar("TModel", bound=nn.Module)


@dataclasses.dataclass(kw_only=True)
class InitializeModelStageResult(Generic[TModel]):
    """
    The result of initializing a model stage.

    Attributes:
        model: The PyTorch module.
        state_mapper: The mapper defining how to load weights into this module.
    """

    model: TModel
    state_mapper: ModelStateMapper


@dataclasses.dataclass(kw_only=True)
class ParallelizeModelStageContext(Generic[TModel]):
    """
    Context data required for horizontally parallelizing a model stage.

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
    """
    Context data required for preparing a model stage for export.

    Attributes:
        dist_context: The distributed execution context.
        model: The PyTorch module to be exported.
    """

    dist_context: DistributedContext
    model: TModel


@dataclasses.dataclass(kw_only=True)
class PrepareExportModelStageResult:
    """
    The result of preparing a model stage for export.

    Attributes:
        state_mapper: The mapper defining how model parameters map to disk storage.
    """

    state_mapper: ModelStateMapper


class ModelProvider(abc.ABC, Generic[TModel]):
    """
    Abstract interface for defining the lifecycle of a distributed model.

    This provider handles initialization, parallelization (sharding/replication/etc), and export preparation
    for models within the d9d framework.
    """

    @abc.abstractmethod
    def initialize_model_stage(
            self,
            context: InitializeModelStageContext
    ) -> InitializeModelStageResult[TModel]:
        """
        Initializes the model architecture for a specific pipeline stage.

        This method is responsible for constructing the `nn.Module` for the requested stage.

        Construction occurs within a meta-device context; therefore, weights
        should not be loaded directly here. Instead, a `ModelStateMapper` must be returned
        to define how weights from a checkpoint map to the newly created module parameters.

        This allows for architecture modifications, such as injecting LoRA adapters,
        provided that the returned mapper reflects the new structure.

        Args:
            context: Context for this operation.

        Returns:
            Result of this operation.
        """

        ...

    @abc.abstractmethod
    def parallelize_model_stage(
            self,
            context: ParallelizeModelStageContext[TModel]
    ):
        """
        Converts the model parameters into distributed tensors (DTensors).

        Implementations should modify the model in-place. This involves converting
        standard parameters into DTensors by replicating or sharding them according
        to the desired parallelism strategies.

        Args:
            context: Context for this operation.
        """

    @abc.abstractmethod
    def prepare_export_model_stage(
            self,
            context: PrepareExportModelStageContext[TModel]
    ) -> PrepareExportModelStageResult:
        """
        Prepares the state mapper required for saving the model to disk.

        This methods defines how the current in-memory model structure maps back to the
        serialized checkpoint format.

        Args:
            context: Context for this operation.

        Returns:
            Result of this operation.
        """

    def dump_hparams(self) -> ScalarTree:
        """
        Exports hyperparameters associated with this model for logging.

        Returns:
            A dictionary of hyperparameter names and values.
        """

        return {}
