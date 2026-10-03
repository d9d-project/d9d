# Data Loading

## About

A `DataProvider` is the factory that you supply to the train or inference loop, like `ModelProvider` or `OptimizerProvider`. Given the run context, it builds a `MicrobatchPackStream`. This is a `Stateful` iterable that yields microbatch packs and reports its length through the `total_steps` property.

## Concepts

*   A **pack** holds the data of one step: a sequence of microbatches. `len(pack)` is the number of microbatches in that step (the gradient accumulation factor). It can vary from step to step. The loop copies packs to the device ahead of their steps (see [Data Prefetching](../train.md#data-prefetching)).
*   **`total_steps`** is the number of steps the stream yields. It is `None` when the length is not known ahead of time, e.g. for streaming or data-dependent batching. `JobSchedule` takes the job duration from `JobScheduleConfig.total_steps` if it is set. It must not exceed the stream length. Otherwise, `JobSchedule` uses the `total_steps` of the stream. The loop runs exactly that many steps. It cuts a longer stream short and raises an error if the stream ends earlier.
*   The stream is the only checkpoint boundary for the data. It saves and restores its own position per data-parallel rank, so the job resumes exactly. Prefetching iterates the stream on a background thread and can call `state_dict()` after every pack. So keep `state_dict()` cheap, and do not return objects that later iteration changes.

You can use the shipped `AutoDataProvider` for the common case, or write your own provider for full control.

## Using `AutoDataProvider`

`AutoDataProvider` builds the default data stack for you. You supply the two parts that cannot be serialized: a `dataset_factory` and a `collator`. An `AutoDataConfig` holds the serializable settings: `global_batch_size`, `microbatch_size`, `shard_indexing_mode`, `drop_last` and loader settings. The loader settings include `shuffle`, `num_workers`, `pin_memory` and `prefetch_factor`. A `PinMemoryMicrobatchPackStream` does the pinning. Unlike the `DataLoader` option, it also pins tensors nested in dataclasses.

`AutoDataProvider` shards the dataset across data-parallel ranks for you. It also builds the loader, derives the gradient accumulation factor and returns the stream. So your factory returns the *unsharded* dataset.

*   The `dataset_factory` receives the `DistributedContext`, so it can guard data preparation. For example, `dist_context.main_process_first()` lets rank 0 fill the cache before the other ranks read it.
*   The `collator` collates a list of samples into one microbatch.

```python
import datasets
import torch

from d9d.loop.auto import AutoDataConfig, AutoDataProvider
from d9d.loop.run import TrainingConfigurator


def build_dataset(dist_context):
    # Rank 0 downloads and processes the data first. The other ranks load from its cache.
    # Return the dataset unsharded: AutoDataProvider shards it across data-parallel ranks.
    with dist_context.main_process_first():
        return datasets.load_dataset("my/dataset", split="train")


def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
    }


provider = AutoDataProvider(
    dataset_factory=build_dataset,
    collator=collate,
    config=AutoDataConfig(
        global_batch_size=256,  # Effective batch across all DP replicas and accumulation
        microbatch_size=4,  # Samples per microbatch on a single rank
        num_workers=4,
        shuffle=True,
    ),
)

# Hand it to the loop alongside the other providers.
trainer = TrainingConfigurator(
    data_provider=provider,
    ...,
).configure()
```

The batch sizes are known, so the stream reports its `total_steps`. `JobSchedule` then derives the job duration without `JobScheduleConfig.total_steps`.

## Writing a Custom `DataProvider`

A custom provider builds the same default stack by hand. It has two layers:

1.  A **loader**: any `DataLoaderProtocol` (from `d9d.core.protocol`). It is a `Stateful`, `Sized` iterable of single collated microbatches. torchdata's `StatefulDataLoader` satisfies it directly.
2.  A **packer**: `FixedCountMicrobatchPacker(microbatches_per_step=k)` groups `k` microbatches into each pack, which gives gradient accumulation. `drop_last` controls whether a short trailing pack is dropped. Training drops it, evaluation keeps it. The helper `num_microbatches_for_global_batch` derives `k`.

Wrap the packer in a `PinMemoryMicrobatchPackStream` to pin the packs, so the loop copies them to the device asynchronously. The packer and the pinning stream live in `d9d.dataset.batch_iterator`.

Unlike with `AutoDataProvider`, you own the data-parallel sharding. Shard the dataset yourself before you build the loader, e.g. with `shard_dataset_data_parallel`. See the [Dataset Utilities](../../dataset/index.md) documentation.

```python
from collections.abc import Sequence
from typing import Any

import torch
import datasets
from pydantic import BaseModel
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from d9d.core.protocol import MicrobatchPackStream
from d9d.core.types import TensorTree
from d9d.dataset import (
    BufferSortedDataset,
    DatasetImplementingSortKeyProtocol,
    FixedCountMicrobatchPacker,
    num_microbatches_for_global_batch,
    shard_dataset_data_parallel,
)
from d9d.loop.control import DataProvider, InitializeDataProviderContext


class ProjectDataset(Dataset, DatasetImplementingSortKeyProtocol):
    def __init__(self, dataset: datasets.Dataset, tokenizer: Tokenizer):
        self._dataset = dataset
        self._tokenizer = tokenizer

    def sort_key(self, index: int) -> Any:
        # Used by BufferSortedDataset to group examples of similar length together.
        return self._dataset[index]["token_counts"]

    def __getitem__(self, index: int) -> TensorTree:
        return {...}

    @classmethod
    def collate(cls, batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {...}

    def __len__(self) -> int:
        return len(self._dataset)


class DataConfig(BaseModel):
    dataset: str
    split: str
    tokenizer: str
    presort_buffer_size: int
    global_batch_size: int
    microbatch_size: int
    num_workers: int


class ProjectDataProvider(DataProvider):
    def __init__(self, config: DataConfig):
        self._config = config

    def __call__(self, context: InitializeDataProviderContext) -> MicrobatchPackStream:
        tokenizer = Tokenizer.from_file(str(self._config.tokenizer))

        # Rank 0 builds the cache first. The other ranks then load from it.
        with context.dist_context.main_process_first():
            data = datasets.load_dataset(self._config.dataset, split=self._config.split)

        dataset = ProjectDataset(data, tokenizer)

        # Length-bucketing buffer, which minimizes padding.
        dataset_buf = BufferSortedDataset(
            dataset,
            buffer_size=self._config.presort_buffer_size,
            pack_size=self._config.global_batch_size,
        )

        # Shard across data-parallel ranks (a custom DataProvider owns sharding).
        dataset_shard = shard_dataset_data_parallel(dataset_buf, context.dist_context)

        # Loader (layer 1): yields one collated microbatch at a time.
        loader = StatefulDataLoader(
            dataset_shard,
            batch_size=self._config.microbatch_size,
            collate_fn=ProjectDataset.collate,
            num_workers=self._config.num_workers,
            drop_last=True,
        )

        # Packer (layer 2): groups microbatches into one pack per step.
        microbatches_per_step = num_microbatches_for_global_batch(
            context.dist_context,
            global_batch_size=self._config.global_batch_size,
            microbatch_size=self._config.microbatch_size,
        )
        return FixedCountMicrobatchPacker(loader, microbatches_per_step=microbatches_per_step, drop_last=True)
```

## API Reference

::: d9d.loop.control.data_provider

::: d9d.loop.auto.auto_data

::: d9d.dataset.batch_iterator
