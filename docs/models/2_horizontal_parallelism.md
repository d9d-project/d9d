---
title: Horizontal Parallelism
---

# Horizontal Parallelism

## About

The `d9d.module.parallelism` package provides high-level strategies for distributing model execution across device meshes.

These strategies are "Horizontal" in the sense that they function within a specific stage of a pipeline (intra-layer parallelism), as opposed to Pipeline Parallelism which is "Vertical" (inter-layer).

## Design

### DTensor-First Architecture

d9d enforces a **DTensor-first** philosophy. We mandate that every trainable parameter in the distributed environment be represented as a `torch.distributed.tensor.DTensor`.

This constraint simplifies the system architecture significantly:

*   **Universal Checkpointing**: The checkpointing engine does not need to know about specific parallel strategies (like "This is DP" or "This is TP"). It simply inspects the `DTensor.placements` attribute to automatically determine how to gather, deduplicate, and save tensors.
*   **Native Synchronization**: Gradient synchronization for replicated parameters is handled entirely by the [d9d internals](../internals/grad_sync.md), that now knows which tensor dimensions are Replicated.

### Composition over Monoliths

We explicitly reject monolithic wrappers like `torch.nn.parallel.DistributedDataParallel` (DDP).

While DDP is efficient for pure Data Parallelism, it acts as a "black box" that assumes ownership of the entire model execution loop.
Instead, d9d relies on **PyTorch's `parallelize_module` API**. This allows for fine-grained, per-submodule parallelism decisions:

*   Layer A can use **Tensor Parallelism** (Row/Col wise).
*   Layer B (e.g., a Router) can use **Replicate Parallelism**.
*   Layer C (e.g., MLP) can use **Expert Parallelism**.

By treating "Data Parallelism" simply as another tiling strategy ("Replicate") within the Tensor Parallel system, we achieve a unified interface for ND parallelism.

## Strategies

### Replicate Parallelism

`parallelize_replicate` implements **Replicate Parallelism**. It replicates parameters across the mesh. Used for Data Parallelism or Context Parallelism.

During the forward pass, it installs hooks that temporarily "unwrap" `DTensor` parameters into standard, local `torch.Tensor` objects. This allows standard PyTorch operations and custom kernels to run without modification, while accessing module's state dict and parameters still yields `DTensor` objects.

### Expert Parallelism (MoE)

Mixture of Experts (MoE) requires a unique parallel strategy where:
1.  **Experts** are sharded across the `ep_shard` mesh dimension (each GPU holds a subset of experts), optionally replicating along `ep_replicate` .
2.  **Routers** are replicated (all GPUs have the same routing logic).

`parallelize_expert_parallel` applies sharding to `MoELayer` modules. It shards the `GroupedLinear` weights along the expert dimension. Simultaneously, it effectively applies `parallelize_replicate` to the router.

### Fully Sharded Data Parallel (FSDP)

`parallelize_fsdp` provides a thin wrapper around PyTorch's native `fully_shard`.

**Difference from standard FSDP:**

* Standard FSDP averages gradients across the mesh (Sum / WorldSize) by default. d9d's wrapper forces the gradients being *summed* rather than *averaged*. This is required for our gradient accumulation logic that is handled externally.
* `parallelize_fsdp` strictly requires a 1D DeviceMesh. To use it in multi-dimensional meshes (e.g., combining Replication and Sharding), use `parallelize_hsdp` or apply `parallelize_replicate` to the other dimensions manually first.

### Hybrid Sharded Data Parallel (HSDP)

`parallelize_hsdp` is a high-level composite strategy for mixing Full Sharding with Replicate Parallel.

`parallelize_hsdp` accepts a multi-dimensional mesh and a target `shard_dim`. It identifies all dimensions *other than* `shard_dim` as **Replication Dimensions**. 
It applies `parallelize_replicate` to the replication dimensions.
It applies `parallelize_fsdp` to the specific sharding dimension.


## Usage Examples

### Replicate Parallelism

```python
import torch
from d9d.core.dist_context import DistributedContext, DENSE_DOMAIN
from d9d.module.parallelism.api import parallelize_replicate

# 1. Create a Distributed Context
ctx: DistributedContext = ...

# 2. Get Dense Domain Mesh
dense_mesh = ctx.mesh_for(DENSE_DOMAIN)  # pp x dp_replicate x dp_cp_shard x cp_replicate x tp

# 2. Define Model
model = MyCustomLayer(...)

# 3. Parallelize
parallelize_replicate(model, dense_mesh[['dp_replicate', 'cp_replicate']])
```

### Applying Expert Parallelism

```python
import torch
from d9d.core.dist_context import DistributedContext, EXPERT_DOMAIN
from d9d.module.parallelism.api import parallelize_expert_parallel
from d9d.module.block.moe import MoELayer

# 1. Create a Distributed Context
ctx: DistributedContext = ...

# 2. Get Expert Domain Mesh
expert_mesh = ctx.mesh_for(EXPERT_DOMAIN)  # pp x ep_replicate x ep_shard

# 3. Define Model
model = MoELayer(...)

# 4. Parallelize
parallelize_expert_parallel(
    model, 
    mesh_experts=expert_mesh[['ep_replicate', 'ep_shard']],
    expert_shard_dim='ep_shard'
)
```

### Applying FSDP

```python
import torch
from d9d.core.dist_context import DistributedContext, DENSE_DOMAIN
from d9d.module.parallelism.api import parallelize_fsdp, parallelize_replicate

# 1. Create a Distributed Context
ctx: DistributedContext = ...

# 2. Define Model
model = MyCustomLayer(...)

# 3. Get Dense Domain Mesh

dense_mesh = ctx.mesh_for(DENSE_DOMAIN)

# 4. Parallelize

parallelize_fsdp(
    model, 
    mesh=dense_mesh['dp_cp_shard']
)
```

### Applying HSDP
```python
import torch
from d9d.core.dist_context import DistributedContext, DENSE_DOMAIN
from d9d.module.parallelism.api import parallelize_hsdp

# 1. Create a Distributed Context
ctx: DistributedContext = ...

# 2. Get Mesh
dense_mesh = ctx.mesh_for(DENSE_DOMAIN)  # pp x dp_replicate x dp_cp_shard x cp_replicate x tp

# 3. Define Model
model = MyCustomLayer(...)

# 4. Parallelize
parallelize_hsdp(
    model,
    mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
    shard_dim="dp_cp_shard"
)
```

::: d9d.module.parallelism.api
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.module.parallelism.style
    options:
        show_root_heading: true
        show_root_full_path: true
