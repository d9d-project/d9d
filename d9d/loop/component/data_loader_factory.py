from collections.abc import Callable, Iterable, Iterator
from typing import Any, Self, TypedDict, Unpack

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from d9d.core.dist_context import BATCH_DOMAIN, DistributedContext
from d9d.core.types import CollateFn, PyTree
from d9d.loop.config import DataLoadingConfig
from d9d.loop.control import DatasetProvider, InitializeDatasetContext

from .batch_maths import BatchMaths


class DataLoaderKwargs(TypedDict, total=False):
    """
    Type definition for arguments accepted by the PyTorch DataLoader.
    """

    batch_size: int | None
    shuffle: bool | None
    sampler: Sampler | Iterable | None
    batch_sampler: Sampler[list] | Iterable[list] | None
    num_workers: int
    collate_fn: CollateFn
    pin_memory: bool
    drop_last: bool
    timeout: float
    worker_init_fn: Callable | None
    multiprocessing_context: Any
    generator: Any
    prefetch_factor: int | None
    persistent_workers: bool
    pin_memory_device: str


def _move_to_device(data: PyTree, device: torch.types.Device) -> PyTree:
    return pytree.tree_map(lambda x: x.to(device), data)


class IteratorBatchGroup(Iterator):
    """
    An iterator that groups items from a base iterator into sub-streams.

    This class is utilized for gradient accumulation where
        a single optimizer step consumes multiple micro-batches (the group).

    It also moves the data to the specified device immediately upon access.
    """

    def __init__(
            self,
            base: Iterator,
            device: torch.types.Device,
            batch_group_size: int
    ):
        """
        Constructs an IteratorBatchGroup object.

        Args:
            base: The underlying data iterator (usually from a DataLoader).
            device: The target device to move tensors to.
            batch_group_size: The number of micro-batches to yield within one group.
        """

        self._base = base
        self._device = device

        self._batch_group_size = batch_group_size

        self._is_end = False

    def __next__(self) -> PyTree:
        """
        Advances the iterator.

        Returns:
            A generator that yields `batch_group_size` items (micro-batches),
            with each item already moved to the configured device.

        Raises:
            StopIteration: If the underlying iterator is exhausted.
        """

        if self._is_end:
            raise StopIteration()

        try:
            sample_item = next(self._base)
        except StopIteration:
            self._is_end = True
            raise StopIteration() from None

        def _iter_inside_group():
            yield _move_to_device(sample_item, self._device)

            for _ in range(self._batch_group_size - 1):
                try:
                    item = next(self._base)
                    yield _move_to_device(item, self._device)
                except StopIteration:
                    self._is_end = True
                    break

        return _iter_inside_group()

    def __iter__(self) -> Self:
        """Returns self."""

        return self


class StatefulDataLoaderDataParallelAware(StatefulDataLoader):
    """
    A stateful data loader that is aware of data parallel ranks.

    This loader extends the standard torchdata StatefulDataLoader to ensure
    that checkpoints are saved with rank-specific keys.

    It also wraps the iterator to support batch grouping for gradient accumulation and
    automatically transfer data to bound device.
    """

    def __init__(
            self,
            dataset: Dataset,
            dp_rank: int,
            device: torch.types.Device,
            group_size: int,
            **kwargs: Unpack[DataLoaderKwargs]
    ):
        """
        Constructs a StatefulDataLoaderDataParallelAware object.

        Args:
            dataset: The dataset to load from.
            dp_rank: The Data Parallel rank of the current process (used for state checkpointing).
            device: The device to move data to.
            group_size: The number of batches to group together (e.g., for gradient accumulation).
            **kwargs: Standard arguments passed to the parent DataLoader.
        """

        super().__init__(dataset, **kwargs)
        self._dp_rank = dp_rank
        self._device = device
        self._group_size = group_size

    def state_dict(self) -> dict[str, Any]:
        return {
            f"dp_{self._dp_rank}": super().state_dict()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict[f"dp_{self._dp_rank}"])

    def __iter__(self) -> Iterator:
        return IteratorBatchGroup(
            super().__iter__(),
            device=self._device,
            batch_group_size=self._group_size
        )


class DataLoaderFactory:
    """
    Factory class for creating configured DataLoaders.

    This class centralizes the creation logic for training and inference
    data loaders, applying configurations.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            provider: DatasetProvider,
            config_data_loading: DataLoadingConfig,
            batch_maths: BatchMaths
    ):
        """
        Constructs a DataLoaderFactory object.

        Args:
            dist_context: The distributed context containing mesh and rank information.
            provider: The provider callable that initializes the dataset and collator.
            config_data_loading: Specific configuration for data loading.
            batch_maths: BatchMaths object.
        """

        self._dist_context = dist_context
        self._provider = provider

        self._config_data_loading = config_data_loading

        self._batch_maths = batch_maths

    def _build_dataloader(
            self,
            provider: DatasetProvider,
            batch_size: int,
            group_size: int,
            drop_last: bool
    ) -> StatefulDataLoader:
        result = provider(InitializeDatasetContext(
            dist_context=self._dist_context,
            batch_maths=self._batch_maths
        ))

        return StatefulDataLoaderDataParallelAware(
            result.dataset,
            collate_fn=result.collator,
            group_size=group_size,
            num_workers=self._config_data_loading.num_workers,
            persistent_workers=self._config_data_loading.persistent_workers,
            pin_memory=self._config_data_loading.pin_memory,
            batch_size=batch_size,
            dp_rank=self._dist_context.mesh_for(BATCH_DOMAIN)["dp"].size(),
            device="cuda",
            drop_last=drop_last
        )

    def build_dataloader_for_train_job(self) -> StatefulDataLoader:
        """
        Builds and returns a StatefulDataLoader configured for training.

        This loader is configured to drop the last incomplete batch and group
        batches according to the gradient accumulation settings defined in
        BatchMaths.

        Returns:
            A configured StatefulDataLoader instance.
        """

        return self._build_dataloader(
            self._provider,
            batch_size=self._batch_maths.data_loader_batch_size,
            group_size=self._batch_maths.num_microbatches_gradient_accumulation,
            drop_last=True
        )

    def build_dataloader_for_infer_job(self) -> StatefulDataLoader:
        """
        Builds and returns a StatefulDataLoader configured for inference.

        This loader processes batches one by one (group size of 1) and does
        not drop the last batch.

        Returns:
            A configured StatefulDataLoader instance.
        """

        return self._build_dataloader(
            self._provider,
            batch_size=self._batch_maths.data_loader_batch_size,
            group_size=1,
            drop_last=False
        )
