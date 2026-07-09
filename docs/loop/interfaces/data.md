# Data Loading

## Concepts

The `DataProvider` is the factory you supply to the train/eval loop — exactly like `ModelProvider` or
`OptimizerProvider`. Given the run context, it composes and returns a **`MicrobatchPackStream`**: a
`Stateful` iterable that yields **microbatch packs** and reports its length via a `total_steps` property.

- A **pack** is one step's worth of data: a sequence of microbatches. `len(pack)` is the number of
  microbatches within that step (the gradient-accumulation factor) and may vary from step to step. The
  loop moves each pack to the device and hands it to the task operator.
- **`total_steps`** is the number of steps the stream will yield, or `None` when that cannot be known
  ahead of time (streaming / data-dependent batching). `JobSchedule` uses it to resolve the job
  duration, falling back to `JobScheduleConfig.total_steps` when it is `None`.
- The stream is the single **checkpoint boundary** for the data: it saves and restores its own position
  (per data-parallel rank) so resumption is exact.

There are two ways to obtain a `DataProvider`: use the shipped `AutoDataProvider` for the common case, or
write your own for full control.

## Using `AutoDataProvider`

`AutoDataProvider` wires the default stack for
you. You supply only the two non-serializable pieces — a `dataset_factory` and a `collator` — plus an
`AutoDataConfig` for the serializable knobs (`global_batch_size`, `microbatch_size`,
`shard_indexing_mode`, `drop_last`, and `DataLoader` settings such as `shuffle` / `num_workers` /
`pin_memory` / `prefetch_factor`).

It **shards the dataset across data-parallel ranks for you**, builds the loader, derives the
gradient-accumulation factor, and returns the stream — so your factory returns the *unsharded* dataset.

- The `dataset_factory` receives the `DistributedContext`, so it can guard data preparation (e.g. with
  `dist_context.main_process_first()` so rank 0 populates the cache before the others read it).
- The `collator` collates a list of samples into one microbatch.

```python
import datasets
import torch

from d9d.loop.auto import AutoDataConfig, AutoDataProvider
from d9d.loop.run import TrainingConfigurator


def build_dataset(dist_context):
    # Only rank 0 downloads/processes; the others load from the cache it builds. Return it UNSHARDED —
    # AutoDataProvider shards across data-parallel ranks itself.
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
        global_batch_size=256,  # effective batch across all DP replicas and accumulation
        microbatch_size=4,  # samples per microbatch on a single rank
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

Because the batch sizes are known, the resulting stream is length-aware (`total_steps` is populated), so
`JobSchedule` derives the job duration without you setting `JobScheduleConfig.total_steps`.

## Writing a custom `DataProvider`

A custom provider composes the same default stack by hand, which is two layers (both in
`d9d.dataset.batch_iterator`):

1. A **loader** — any `DataLoaderProtocol`: a `Stateful`, `Sized` iterable of single collated
   microbatches. torchdata's `StatefulDataLoader` satisfies it directly.
2. A **packer** — `FixedCountMicrobatchPacker(microbatches_per_step=k)` groups `k` microbatches into each
   pack, reproducing gradient accumulation (`drop_last` controls whether a short trailing pack is
   dropped — training drops it, evaluation keeps it). The accumulation factor `k` is derived with the
   helper `num_microbatches_for_global_batch`.

Unlike `AutoDataProvider`, **you own the data-parallel sharding**: shard the dataset yourself (e.g. with
`shard_dataset_data_parallel`) before building
the loader. See the [Dataset Utilities](../dataset/index.md) documentation.

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

        # main_process_first ensures rank 0 builds the cache first; other ranks then load from it.
        with context.dist_context.main_process_first():
            data = datasets.load_dataset(self._config.dataset, split=self._config.split)

        dataset = ProjectDataset(data, tokenizer)

        # Length-bucketing buffer (minimizes padding overhead).
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

## API reference

::: d9d.loop.control.data_provider

::: d9d.loop.auto.auto_data

::: d9d.dataset.batch_iterator
