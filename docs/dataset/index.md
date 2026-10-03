# Datasets

## About

The `d9d.dataset` package provides PyTorch `Dataset` wrappers and helpers for distributed training. It covers length-based bucketing, data-parallel sharding, padding and token pooling masks.

## Why d9d Does Not Wrap Datasets Automatically

d9d provides explicit wrappers that you compose yourself. It does not inject samplers or wrap datasets behind your back, as some other frameworks do.

*   **Order of composition**: A data pipeline behaves differently depending on the order of its wrappers. For example, bucketing before sharding is not the same as sharding before bucketing. When you stack the wrappers yourself, you control this order.
*   **Per-dataset configuration**: Datasets have different physical limits. A dataset on network storage can need contiguous reads (`ShardIndexingMode.chunked`), while an in-memory dataset can use round-robin access (`ShardIndexingMode.sequential`). Explicit wrappers expose these options instead of hiding them in global trainer arguments.

## Bucketing

In sequence processing, the items of a batch often have different lengths. With random sampling, every batch is padded to its longest sequence, so compute is spent on padding tokens.

`BufferSortedDataset` reads a buffer of items, sorts it by a sort key (usually the length) and splits it into packs. Items in one pack, e.g. one microbatch, have similar lengths, which reduces padding. It then shuffles the packs and the items within each pack, so the data order is not strictly sorted.

The underlying dataset must implement `DatasetImplementingSortKeyProtocol`, i.e. it must have a `sort_key(index)` method.

## Sharding

With data parallelism, each GPU processes a subset of the data. `ShardedDataset` is a deterministic view of one shard of the data, selected by the shard index (usually the data-parallel rank).

It supports:

*   **Sequential sharding**: Round-robin distribution (`0, 4, 8, ...` for rank 0 of 4).
*   **Chunked sharding**: Contiguous blocks (`0, 1, 2, ...` for rank 0).
*   **Optional padding**: All shards get the same length by repeating the last item. Without it, ranks with fewer items finish early, and the other ranks hang in collectives.

`shard_dataset_data_parallel` builds a `ShardedDataset` for the data-parallel dimension of a `DistributedContext`.

A `ShardedDataset` must be restored from a checkpoint with the same number of shards.

## Padding

Variable-length sequences must be padded to the same length before they can be stacked into a batch. `pad_stack_1d` pads and stacks 1D tensors. Use it to write a `collate_fn`.

## Token Pooling

`token_pooling_mask_from_attention_mask` builds a mask of the tokens to pool from an attention mask. `TokenPoolingType` selects the first token, the last non-padding token or all non-padding tokens. The `last` mode assumes right padding.

## Microbatch Packing

`FixedCountMicrobatchPacker`, `PinMemoryMicrobatchPackStream` and `num_microbatches_for_global_batch` build the stream of microbatch packs that the training loop consumes. See [Data Loading](../loop/interfaces/data.md) for how to use them.

## Usage

### Bucketing

```python
from torch.utils.data import Dataset
from d9d.dataset import BufferSortedDataset

class MyTextDataset(Dataset):
    def __init__(self, data: list[str]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    # Required by BufferSortedDataset.
    def sort_key(self, index):
        return len(self.data[index])

# Shuffle the base dataset globally beforehand if possible.
raw_data = ["short", "very very very long phrase", "tiny", "medium size"] * 100
base_ds = MyTextDataset(raw_data)

# - buffer_size=100: sort 100 items at a time to find similar lengths
# - pack_size=4: group them into batches of 4.
sorted_ds = BufferSortedDataset(
    base_dataset=base_ds,
    buffer_size=100,
    pack_size=4,
    init_seed=42
)
```

### Sharding for Data Parallelism

```python
import torch
from torch.utils.data import TensorDataset
from d9d.core.dist_context import DistributedContext
from d9d.dataset import shard_dataset_data_parallel, ShardIndexingMode

# The data-parallel size and rank come from the DistributedContext.
context: DistributedContext

base_ds = TensorDataset(torch.randn(100, 10))

sharded_ds = shard_dataset_data_parallel(
    dataset=base_ds,
    dist_context=context,
    # Optional parameters.
    indexing_mode=ShardIndexingMode.chunked,
    pad_to_equal_size_across_shards=True
)

print(f"I see {len(sharded_ds)} items.")
```

### Manual Sharding

```python
import torch
from torch.utils.data import TensorDataset
from d9d.core.dist_context import DistributedContext, BATCH_DOMAIN
from d9d.dataset import ShardedDataset, ShardIndexingMode

# Take the data-parallel size and rank from the DistributedContext.
context: DistributedContext
dp_mesh = context.mesh_for(BATCH_DOMAIN)["dp"]
dp_size = dp_mesh.size()
dp_rank = dp_mesh.get_local_rank()

base_ds = TensorDataset(torch.randn(100, 10))

sharded_ds = ShardedDataset(
    dataset=base_ds,
    total_shards=dp_size,
    current_shard=dp_rank,
    indexing_mode=ShardIndexingMode.chunked,
    # Prevents hangs in collectives when shards have different lengths.
    pad_to_equal_size_across_shards=True
)

print(f"I am rank {dp_rank}, I see {len(sharded_ds)} items.")
```

### Padding

```python
import torch
from d9d.dataset import pad_stack_1d, PaddingSide1D

items = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4]),
    torch.tensor([5, 6])
]

# 1. Right padding.
batch = pad_stack_1d(items, pad_value=0, padding_side=PaddingSide1D.right)

# 2. Left padding.
batch_left = pad_stack_1d(items, pad_value=0, padding_side=PaddingSide1D.left)

# 3. Padding to a multiple of 8, e.g. for kernel requirements or context-parallel sharding.
batch_aligned = pad_stack_1d(
    items,
    pad_value=0,
    pad_to_multiple_of=8
)
```

## API Reference

::: d9d.dataset
