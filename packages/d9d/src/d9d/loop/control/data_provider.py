"""Interface for providing training and inference data."""

import dataclasses
import typing
from typing import Protocol

from d9d.data import MicrobatchPackStream
from d9d.dist_context import DistributedContext
from d9d.rng import SeedScope


@dataclasses.dataclass(kw_only=True)
class InitializeDataProviderContext:
    """Context data required to initialize a data provider.

    Attributes:
        dist_context: The distributed context containing rank and world size information.
        seed_scope: The seed scope for code before sharding. It is equal on all ranks.
        shard_seed_scope: The seed scope for code after sharding, e.g. the generator of the data loader. It is
            equal on the ranks of one data-parallel shard.
    """

    dist_context: DistributedContext
    seed_scope: SeedScope
    shard_seed_scope: SeedScope


@typing.runtime_checkable
class DataProvider(Protocol):
    """Protocol for defining how the data pipeline is built.

    A ``DataProvider`` is the factory the user supplies to the train/eval loop, exactly like
    ``ModelProvider`` or ``OptimizerProvider``. Given the run context, it composes and returns a
    ``MicrobatchPackStream``.

    The user is responsible for sharding the dataset across data-parallel ranks (e.g. with
    ``d9d.data.shard_dataset_data_parallel``) inside the provider.

    The iteration over the data can run on a prefetch thread, so it must not draw from the global generators.
    It must take its seeds from the seed scopes of the context instead.
    """

    def __call__(self, context: InitializeDataProviderContext) -> MicrobatchPackStream:
        """Builds the microbatch pack stream for the job.

        Args:
            context: Context for this operation.

        Returns:
            The microbatch pack stream the loop will drive.
        """
        ...
