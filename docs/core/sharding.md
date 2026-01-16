---
title: Tensor Sharding
---

# Tensor Sharding

## About

The `d9d.core.sharding` package provides utilities for splitting and reconstructing complex nested structures (PyTrees) of PyTorch Tensors.

## Sharding Spec

A **Sharding Spec** is a PyTree that mirrors the structure of your data (e.g., a State Dict).

*   **Structure**: Must match the data tree exactly (dicts, lists, custom nodes).
*   **Leaves**:
    *   `torch.distributed.tensor.Shard(dim)`: Indicates the tensor at this position should be split along dimension `dim`.
    *   `None`: Indicates the leaf should be replicated (kept as-is/not split).

Helper functions like `shard_on_dim` allow generating these specs automatically.

## Usage Examples

### Basic Split and Join

This example demonstrates how to shard a simple dictionary of tensors across a hypothetical "world size" of 2.

```python
import torch
from d9d.core.sharding import (
    shard_tree,
    unshard_tree,
    shard_spec_on_dim
)

# 1. Define Data
data = {
    "main": {
        "inputs": torch.randn(4, 4),
        "bias": torch.randn(4),
    },
    "scalar": torch.scalar_tensor(0.0),
    "something_else": False
}

# 2. Create Spec
# Automatically mark all tensors (that are not 0-dim) to be split on dimension 0
spec = shard_spec_on_dim(data, dim=0)

# 3. Apply Sharding
# This maps 'main'->'inputs' and 'main'->'bias' into tuples of Tensor-s
sharded_data = shard_tree(
    data,
    spec,
    num_shards=2,
    enforce_even_split=True
)

print(sharded_data)

# 4. Reconstruct
# Concatenates the tuples back into full tensors
reconstructed_data = unshard_tree(sharded_data, spec)

print(reconstructed_data)
```

::: d9d.core.sharding
    options:
        show_root_heading: true
        show_root_full_path: true
