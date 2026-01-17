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

::: d9d.core.sharding
    options:
        show_root_heading: true
        show_root_full_path: true
