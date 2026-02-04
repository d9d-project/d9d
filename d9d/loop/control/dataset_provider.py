import dataclasses
import typing
from typing import Protocol

from torch.utils.data import Dataset

from d9d.core.dist_context import DistributedContext
from d9d.core.types import CollateFn

if typing.TYPE_CHECKING:
    from d9d.loop.component import BatchMaths


@dataclasses.dataclass(kw_only=True)
class InitializeDatasetContext:
    """Context data required to initialize a dataset provider.

    Attributes:
        dist_context: The distributed context containing rank and world size information.
        batch_maths: The batch maths component handling global batch size calculations.
    """

    dist_context: DistributedContext
    batch_maths: "BatchMaths"


@dataclasses.dataclass(kw_only=True)
class InitializeDatasetResult:
    """The result of initializing a dataset provider.

    Attributes:
        dataset: The instantiated PyTorch Dataset.
        collator: The function used to collate individual samples into a batch.
    """

    dataset: Dataset
    collator: CollateFn


@typing.runtime_checkable
class DatasetProvider(Protocol):
    """Protocol that allows users to define how datasets are loaded and collated.

    Users should subclass this to provide custom data loading logic.
    """

    def __call__(self, context: InitializeDatasetContext) -> InitializeDatasetResult:
        """
        Initializes the dataset components.

        It is important that the user must shard the dataset manually, perhaps using `d9d.dataset.ShardedDataset`.

        Args:
            context: Context for this operation.

        Returns:
            Result of this operation.
        """
