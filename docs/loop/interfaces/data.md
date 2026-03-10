# Data Loading

## DatasetProvider

The `DatasetProvider` is responsible for creating dataset and data collator instances.

## Distributed-Awareness

d9d **will not** apply sharding to your dataset automatically. You have to configure it manually (optionally applying other dataset wrappers).

Please see the [Dataset Utilities](../dataset/index.md) documentation.

## Example Implementation

```python
from typing import Any, Sequence

import torch
import datasets
from pydantic import BaseModel
from tokenizers import Tokenizer

from d9d.core.types import TensorTree
from d9d.dataset import BufferSortedDataset, shard_dataset_data_parallel, DatasetImplementingSortKeyProtocol
from d9d.loop.control.dataset_provider import *

class ProjectDataset(Dataset, DatasetImplementingSortKeyProtocol):
    def __init__(self, dataset: datasets.Dataset, tokenizer: Tokenizer):
        self._dataset = dataset
        self._tokenizer = tokenizer

    def sort_key(self, index: int) -> Any:
        # Used by BufferSortedDataset to group examples of similar length together.
        # This minimizes padding overhead in batches.
        return self._dataset[index]["token_counts"]

    def __getitem__(self, index: int) -> TensorTree:
        return {
            ...
        }

    @classmethod
    def collate(cls, batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            ...
        }

    def __len__(self) -> int:
        return len(self._dataset)


class DataConfig(BaseModel):
    dataset: str  # HuggingFace dataset path/name
    split: str  # e.g., 'train', 'validation'
    text_column: str  # The column containing the raw text
    use_samples: int  # Limit dataset size for testing/debugging
    shuffle_seed: int  # Distinct seed for shuffling the data
    tokenizer: str  # Path to the tokenizer.json file
    num_proc: int  # Number of CPU processes for data mapping
    presort_buffer_size: int  # Size of buffer for length-based presorting

    
class ProjectDatasetProvider(DatasetProvider):
    def __init__(self, config: DataConfig):
        self._config = config

    @staticmethod
    def _count_tokens(item: dict, text_column: str, tokenizer: Tokenizer) -> dict:
        return {
            "token_counts": len(tokenizer.encode(item[text_column]).tokens)
        }

    def __call__(self, context: InitializeDatasetContext) -> InitializeDatasetResult:
        tokenizer = Tokenizer.from_file(str(self._config.tokenizer))
        # IMPORTANT: main_process_first ensures that Rank 0 downloads/processes
        # the dataset and builds the cache first. Ranks 1-N wait, then load from cache.
        # Prevents race conditions and corruption on the HF cache.
        with context.dist_context.main_process_first():
            data = datasets.load_dataset(
                self._config.dataset,
                split=self._config.split
            ).take(
                self._config.use_samples
            ).shuffle(
                self._config.shuffle_seed
            ).map(
                self._count_tokens,
                num_proc=self._config.num_proc,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "text_column": self._config.text_column
                }
            )

        dataset = ProjectDataset(data, tokenizer)

        # BufferSortedDataset acts as a buffer that shuffles data locally
        # but outputs batches sorted by length (defined in sort_key above)
        dataset_buf = BufferSortedDataset(
            dataset,
            buffer_size=self._config.presort_buffer_size,
            pack_size=context.batch_maths.global_batch_size,
            init_seed=self._config.shuffle_seed
        )

        # Split dataset across data parallel ranks
        dataset_shard = shard_dataset_data_parallel(dataset_buf, context.dist_context)

        return InitializeDatasetResult(
            dataset=dataset_shard,
            collator=ProjectDataset.collate
        )
```

::: d9d.loop.control.dataset_provider
