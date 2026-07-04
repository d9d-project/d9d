from collections.abc import Callable

from pydantic import BaseModel
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from d9d.core.dist_context import DistributedContext
from d9d.core.protocol import MicrobatchPackStream
from d9d.core.types import CollateFn
from d9d.dataset import (
    FixedCountMicrobatchPacker,
    ShardIndexingMode,
    num_microbatches_for_global_batch,
    shard_dataset_data_parallel,
)
from d9d.loop.control import DataProvider, InitializeDataProviderContext

DatasetFactory = Callable[[DistributedContext], Dataset]
"""A callable that builds the (unsharded) dataset, given the distributed context.

It receives the context so it can guard data preparation (e.g. with ``dist_context.main_process_first()``).
"""


class AutoDataConfig(BaseModel):
    """Configuration for the default data pipeline.

    Attributes:
        global_batch_size: The total effective batch size across all replicas and accumulation.
        microbatch_size: The number of samples in a single microbatch on a single rank.
        shard_indexing_mode: The dataset sharding strategy.
        shuffle: Whether to reshuffle the data every epoch.
        drop_last: Whether to drop the trailing incomplete microbatch and pack (set False for evaluation).
        num_workers: The number of subprocesses to use for data loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory before returning them.
        persistent_workers: Whether to keep worker processes alive between epochs.
        prefetch_factor: The number of batches each worker prefetches ahead.
        timeout: The timeout in seconds for collecting a batch from workers.
    """

    global_batch_size: int
    microbatch_size: int
    shard_indexing_mode: ShardIndexingMode = ShardIndexingMode.sequential
    shuffle: bool = False
    drop_last: bool = True
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    timeout: float = 0.0


class AutoDataProvider(DataProvider):
    """DataProvider that wires the default stack: shard the dataset, load microbatches, pack them per step.

    It shards the dataset across data-parallel ranks, wraps it in a stateful loader at the microbatch size,
    derives the gradient-accumulation factor from the global batch size, and groups the microbatches with a
    ``FixedCountMicrobatchPacker``. Users who need a non-default stack should write their own ``DataProvider``.
    """

    def __init__(self, dataset_factory: DatasetFactory, collator: CollateFn, config: AutoDataConfig):
        """Constructs the AutoDataProvider object.

        Args:
            dataset_factory: Builds the unsharded dataset given the distributed context.
            collator: Collates individual samples into a microbatch.
            config: The serializable settings for the default stack.
        """
        self._dataset_factory = dataset_factory
        self._collator = collator
        self._config = config

    def __call__(self, context: InitializeDataProviderContext) -> MicrobatchPackStream:
        dataset = self._dataset_factory(context.dist_context)
        dataset = shard_dataset_data_parallel(
            dataset, context.dist_context, indexing_mode=self._config.shard_indexing_mode
        )

        loader = StatefulDataLoader(
            dataset,
            batch_size=self._config.microbatch_size,
            collate_fn=self._collator,
            shuffle=self._config.shuffle,
            num_workers=self._config.num_workers,
            pin_memory=self._config.pin_memory,
            persistent_workers=self._config.persistent_workers,
            prefetch_factor=self._config.prefetch_factor,
            timeout=self._config.timeout,
            drop_last=self._config.drop_last,
        )

        microbatches_per_step = num_microbatches_for_global_batch(
            context.dist_context, self._config.global_batch_size, self._config.microbatch_size
        )

        return FixedCountMicrobatchPacker(
            loader, microbatches_per_step=microbatches_per_step, drop_last=self._config.drop_last
        )
