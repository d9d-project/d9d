---
title: Datasets
---

# Datasets

## About

The `d9d.dataset` package provides specialized PyTorch `Dataset` wrappers designed for distributed training scenarios.

## Core Concepts

### Why Not Auto-Wrap Datasets Automatically?

d9d provides explicit composable wrappers rather than relying on implicit "magic" or automatic Sampler injection often found in other frameworks.

* **Flexible Composition and Order-of-Operations**: The behavior of a data pipeline changes significantly depending on the order of composition. By stacking wrappers manually, you control the data flow logic:

* **Granular Configuration**: Different datasets have different physical constraints that require specific configurations. A dataset loaded from network storage might require contiguous reads to be performant (`ShardIndexingMode.chunked`), while an in-memory dataset might prefer round-robin access (`ShardIndexingMode.sequential`). Explicit wrappers ensure that these configuration options are exposed to the user rather than buried in global trainer arguments.


## Features

### Smart Bucketing

In NLP and Sequence processing, batches often contain items of varying lengths. Standard random sampling forces the batch to be padded to the length of the longest sequence, wasting computational resources on padding tokens.

`BufferSortedDataset` implements a "Smart Bucketing" strategy to balance efficiency and statistical variance. It ensures that items within a specific micro-batch have similar lengths (minimizing padding), while preventing the data stream from becoming strictly deterministic or sorted.

#### Usage Example

To use `BufferSortedDataset`, your underlying dataset must implement the `DatasetImplementingSortKeyProtocol` (i.e., it must have a `sort_key(index)` method).

```python
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from d9d.dataset import BufferSortedDataset

class MyTextDataset(Dataset):
    def __init__(self, data: list[str]):
        self.data = data # list of strings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    # ! You need to implement this one \/ !
    def sort_key(self, index):
        return len(self.data[index])

# Create Base Dataset (Ideally globally shuffled beforehand)
raw_data = ["short", "very very very long phrase", "tiny", "medium size"] * 100
base_ds = MyTextDataset(raw_data)

# Wrap with Smart Bucketing
# - buffer_size=100: Look at 100 items at a time to find similar lengths
# - pack_size=4: Group them into batches of 4
sorted_ds = BufferSortedDataset(
    base_dataset=base_ds,
    buffer_size=100,
    pack_size=4,
    init_seed=42
)
```

### Sharding

When using Data Parallelism, each GPU processes a subset of the data. `ShardedDataset` provides a deterministic view of a specific shard of the data based on the rank (shard ID).

It supports:

*   **Sequential Sharding**: Round-robin distribution (`0, 4, 8...` for rank 0).
*   **Chunked Sharding**: Contiguous blocks (`0, 1, 2...` for rank 0).
*   **Optional Padding**: Ensuring all shards have exactly the same length. This is critical for distributed training loops where uneven dataset sizes can cause process hangs.

#### Usage Example (for Data Parallel)

```python
import torch
from torch.utils.data import TensorDataset
from d9d.core.dist_context import DistributedContext, BATCH_DOMAIN
from d9d.dataset import shard_dataset_data_parallel, ShardIndexingMode

# You can infer your Data Parallel size and rank from DistributedContext object 
context: DistributedContext

# Create Full Dataset
base_ds = TensorDataset(torch.randn(100, 10))

# Shard it
sharded_ds = shard_dataset_data_parallel(
    dataset=base_ds,
    dist_context=context,
    # Optional Parameters:
    indexing_mode=ShardIndexingMode.chunked,
    pad_to_equal_size_across_shards=True 
)

print(f"I am rank {dp_rank}, I see {len(sharded_ds)} items.")
```

#### Usage Example (Manual)

```python
import torch
from torch.utils.data import TensorDataset
from d9d.core.dist_context import DistributedContext, BATCH_DOMAIN
from d9d.dataset import ShardedDataset, ShardIndexingMode

# You can infer your Data Parallel size and rank from DistributedContext object 
context: DistributedContext
batch_mesh = context.mesh_for(BATCH_DOMAIN)
dp_size = batch_mesh.size('dp')
dp_rank = batch_mesh.get_local_rank('dp')

# Create Full Dataset
base_ds = TensorDataset(torch.randn(100, 10))

# Shard it
sharded_ds = ShardedDataset(
    dataset=base_ds,
    total_shards=dp_size,
    current_shard=dp_rank,
    indexing_mode=ShardIndexingMode.chunked,
    # Crucial for preventing distributed hangs
    pad_to_equal_size_across_shards=True 
)

print(f"I am rank {dp_rank}, I see {len(sharded_ds)} items.")
```

### Padding Utilities

When creating batches from variable-length sequences, tensors must be padded to the same length to form a valid tensor stack.

`pad_stack_1d` provides a robust utility for this, specifically designed to help writing `collate_fn`.

#### Usage Example

```python
import torch
from d9d.dataset import pad_stack_1d, PaddingSide1D

# Variable length sequences
items = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4]),
    torch.tensor([5, 6])
]

# 1. Standard Right Padding
batch = pad_stack_1d(items, pad_value=0, padding_side=PaddingSide1D.right)

# 2. Left Padding 
batch_gen = pad_stack_1d(items, pad_value=0, padding_side=PaddingSide1D.left)

# 3. Aligned Padding
# Ensures the dimensions are friendly to GPU kernels or for Context Parallel sharding
batch_aligned = pad_stack_1d(
    items, 
    pad_value=0, 
    pad_to_multiple_of=8
)
```

::: d9d.dataset
    options:
        show_root_heading: true
        show_root_full_path: true
