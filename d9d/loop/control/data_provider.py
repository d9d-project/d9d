import dataclasses
import typing
from typing import Protocol

from d9d.core.dist_context import DistributedContext
from d9d.core.protocol import MicrobatchPackStream


@dataclasses.dataclass(kw_only=True)
class InitializeDataProviderContext:
    """Context data required to initialize a data provider.

    Attributes:
        dist_context: The distributed context containing rank and world size information.
    """

    dist_context: DistributedContext


@typing.runtime_checkable
class DataProvider(Protocol):
    """Protocol that allows users to define how the data pipeline is built.

    A ``DataProvider`` is the factory the user supplies to the train/eval loop, exactly like
    ``ModelProvider`` or ``OptimizerProvider``. Given the run context, it composes and returns a
    ``MicrobatchPackStream``.

    The user is responsible for sharding the dataset across data-parallel ranks (e.g. with
    ``d9d.dataset.shard_dataset_data_parallel``) inside the provider.
    """

    def __call__(self, context: InitializeDataProviderContext) -> MicrobatchPackStream:
        """Builds the microbatch pack stream for the job.

        Args:
            context: Context for this operation.

        Returns:
            The microbatch pack stream the loop will drive.
        """
        ...
