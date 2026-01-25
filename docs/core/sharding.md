---
title: Tensor Sharding
---

# Tensor Sharding

## About
The `d9d.core.sharding` package provides utilities for splitting and reconstructing complex nested structures (PyTrees) of PyTorch Tensors and Python Lists.

## Sharding Spec

A **Sharding Spec** is a PyTree that mirrors the structure of your data (e.g., a State Dict).

*   **Structure**: Mirrors the data hierarchy. The spec structure is used to traverse the data; sharding operations flatten the data tree *up to* the leaves defined in the spec.
*   **Leaves**:
    *   `torch.distributed.tensor.Shard(dim)`:
        *   **Tensors**: The tensor is split along dimension `dim`.
        *   **Lists**: The list is split into chunks. `dim` must be `0`.
    *   `None`: The item is replicated (kept as-is/not split) across all shards.

Helper functions like `shard_spec_on_dim` allow generating these specs automatically.

::: d9d.core.sharding
    options:
        show_root_heading: true
        show_root_full_path: true
